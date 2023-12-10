import torch
import torch.nn as nn
import torch.nn.functional as F

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.optim import build_optimizer, build_lr_scheduler
from torch.cuda.amp import GradScaler, autocast

from clip import clip
from clip.model import convert_weights

from .coop import load_clip_to_cpu
from .imagenet_templates import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from dassl.utils import load_pretrained_weights, load_checkpoint
import math
from dassl.metrics import compute_accuracy
_tokenizer = _Tokenizer()

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model




def graph_norm_ours(A, batch=False, self_loop=True, symmetric=True):
	# A = A + I    A: (bs, num_nodes, num_nodes
    # Degree
    d = A.sum(-1) # (bs, num_nodes) #[1000, m+1]
    if symmetric:
		# D = D^-1/2
        d = torch.pow(d, -0.5)
        if batch:
            D = A.detach().clone()
            for i in range(A.size(0)):
                D[i] = torch.diag(d[i])
            norm_A = D.bmm(A).bmm(D)
        else:
            D = torch.diag(d)
            norm_A = D.mm(A).mm(D)
    else:
		# D=D^-1
        d = torch.pow(d,-1)
        if batch:
            D = A.detach().clone()
            for i in range(A.size(0)):
                D[i] = torch.diag(d[i])
            norm_A = D.bmm(A)
        else:
            D =torch.diag(d)
            norm_A = D.mm(A)

    return norm_A

def cal_similarity(x, p=2, dim=1):
    '''
    x: (n,K)
    return: (n,n)
    '''
    x = F.normalize(x, p=p, dim=dim)
    return torch.mm(x, x.transpose(0, 1))

def cal_edge_emb(x, p=2, dim=1):   # v1_graph---taking the similairty by 
    ''' 
    x: (n,K)   [m+1, 1000, 1024]
    return: (n^2, K)
    '''
    x = F.normalize(x, p=p, dim=dim)    #[m+1, 1000, 1024], [100, 1024, 101]
    x_c = x
    x = x.transpose(1, 2)  #[1000, m+1, 1024]  [100, 101, 1024]
    x_r = x  # (K, n, 1) #[1000, m+1, 1024]
    # x_c = torch.transpose(x, 1, 2)  # (K, 1, n) #[1000, 1024, m+1]
    # A = torch.bmm(x_r, x_c).permute(1,2,0)  # (n, n, K) 
    A = torch.bmm(x_r, x_c)     # [1000, m+1, m+1]

    # A = A.view(A.size(0) * A.size(1), A.size(2))  # (n^2, K)
    # print(A.size())
    return A


class GraphConvolution(nn.Module):
    def __init__(self, hidden_dim, name=None, device=None, class_num=None, sparse_inputs=False, act=nn.Tanh, bias=True, dropout=0.0):
        super().__init__()
        self.act = nn.Tanh()
        self.device=device
        self.dropout = dropout
        self.sparse_inputs = sparse_inputs
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.hidden_dim = 512
        self.class_num = class_num
        self.gcn_weights = nn.Parameter(torch.ones(self.hidden_dim, self.hidden_dim))
        if self.bias:
            self.gcn_bias = nn.Parameter(torch.zeros(class_num, self.hidden_dim))
           
        # else:
        #     self.register_parameter('bias', None)
        self.reset_parameters()
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.gcn_weights.size(1))
        self.gcn_weights.data.uniform_(-stdv, stdv)
        # if self.bias is not None:
        #     self.gcn_bias.data.uniform_(-stdv, stdv)

    def forward(self, feat, adj):
        x = feat        #[100, 1024, 101]
        node_size = adj.size()[1]  
        adj = torch.clip(adj, min=0.0)
        I = torch.eye(node_size, device='cuda').unsqueeze(dim=0).to(self.device)
        adj = adj + I      # [1000, m+1, m+1]
        adj = graph_norm_ours(adj, batch=True, self_loop=True, symmetric=True)  #[1000, m+1, m+1]
        x = x.transpose(1, 2)
        pre_sup = torch.matmul(x, self.gcn_weights)  # [m+1, 1000, 1024]
        output = torch.matmul(adj, pre_sup) #[1000, m+1, 1024]

        if self.bias:
            output += self.gcn_bias.unsqueeze(1)
        if self.act is not None:
            return self.act(output[:, 0, :])
        else:
            return output[:, 0, :]


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x



class GraphLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, base_text_features, base_img_features):
        super().__init__()
        self.device = clip_model.dtype
        # self.alpha = cfg.TRAINER.COOP.RESIDUAL_SCALE
        self.alpha=0.1
        print(">> DCT scale factor: ", self.alpha)
        self.register_buffer("base_text_features", base_text_features) #[1000, 1024]
        self.register_buffer("base_img_features", base_img_features)
        # self.alpha_it = cfg.TRAINER.GRAPHADAPTER.ALPHA
        self.alpha_it = 0.7
        self.beta_it = cfg.TRAINER.GRAPHADAPTER.BETA
        self.node_num = 1
        # self.alpha_it = 
        self.hidden_dim = 1
        self.GCN_tt = GraphConvolution(self.hidden_dim, name='metagraph', device=self.device, class_num=base_text_features.size()[0])
        self.GCN_it = GraphConvolution(self.hidden_dim, name='metagraph', device=self.device, class_num=base_text_features.size()[0])

    def reset_parameters(self):
        for i in range(self.node_num):
            stdv = 1. / math.sqrt(self.graph_node[i].size(0))
            self.graph_node[i].data.uniform_(-stdv, stdv)

    def forward(self, img_feature):
        sigma=2.0

        with torch.no_grad():
            node_cluster_t = self.base_text_features.view(1, self.base_text_features.size()[0]//4, 4, self.base_text_features.size()[1])
            node_cluster_i = self.base_img_features.view(1, self.base_img_features.size()[0]//4, 4, self.base_img_features.size()[1])
           
        graph_o_t_all = []
            
        for index in range(4):
            # print("========index", index)
            with torch.no_grad():
                inputs_text = self.base_text_features.unsqueeze(dim=1)    #[100, 1, 1024]
                inputs_img = img_feature.unsqueeze(dim=1)
                node_cluster_tt =  node_cluster_t[:, :, index, :].repeat(inputs_text.size()[0], 1, 1)  #[100, 100, 1024] t->t
                node_cluster_it =  node_cluster_i[:, :, index, :].repeat(inputs_text.size()[0], 1, 1)  # i -> t
                feat_tt = torch.cat([inputs_text, node_cluster_tt], dim=1) 
                feat_it = torch.cat([inputs_text, node_cluster_it], dim=1) 
                feat_tt = feat_tt.transpose(1, 2).detach()
                feat_it = feat_it.transpose(1, 2).detach()
                edge_tt = cal_edge_emb(feat_tt).detach()
                edge_it = cal_edge_emb(feat_it).detach()
            graph_o_tt = self.GCN_tt(feat_tt, edge_tt)
            graph_o_it = self.GCN_it(feat_it, edge_it)
            graph_o_t = (graph_o_tt)*self.alpha_it + (1-self.alpha_it)*graph_o_it
            graph_o_t_all.append(graph_o_t)
        graph_o_t = torch.stack(graph_o_t_all, dim=0).mean(dim=0)
    
        return self.beta_it * self.base_text_features + (1-self.beta_it) * graph_o_t.squeeze(), img_feature



def _get_base_image_features(cfg, classnames, clip_model, img_encoder, train_loader_x):
    device = next(img_encoder.parameters()).device
    if clip_model.dtype == torch.float16:
        img_encoder = img_encoder.cuda()
    # text_encoder = text_encoder.cuda()
    # dataset = cfg.DATASET.NAME
    with torch.no_grad():
        img_feature = []
        labels = []
        for epch in range(10):
            for batch_idx, batch in enumerate(train_loader_x):
                image = batch["img"]
                label = batch["label"]
                image = image.cuda()
                label = label.cuda()
                image_features = img_encoder(image.type(clip_model.dtype)).detach()
                img_feature.append(image_features)
                labels.append(label)
        img_feature_list = torch.cat(img_feature, dim=0)
        label_list = torch.cat(labels, dim=0)
        sorted, indices = torch.sort(label_list)
        # print("+++++++++++++++++++len_label", len(sorted), sorted[-1])
        label_len = len(sorted)//(sorted[-1]+1)
        # print('=====label_len', label_len)
        img_feature_list_all = torch.index_select(img_feature_list, 0, indices)
        b, c = img_feature_list_all.size()
        label_list = sorted.view(b//label_len, label_len)
        img_feature_list_all = img_feature_list_all.view(b//label_len, label_len, -1).mean(dim=1)
        img_encoder = img_encoder.to(device)

    return img_feature_list_all.to(device)





class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, train_loader_x):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype   # float16
        text_encoder = TextEncoder(clip_model)
        img_encoder = self.image_encoder
        base_text_features = _get_base_text_features(cfg, classnames, clip_model, text_encoder)
        base_img_features = _get_base_image_features(cfg, classnames, clip_model, img_encoder, train_loader_x)
       

        self.graph_learner = GraphLearner(cfg, classnames, clip_model, base_text_features, base_img_features)

    def forward(self, image):
        try:
            image_features = self.image_encoder(image.type(self.dtype)).detach()
        except:
            image_features = self.image_encoder(image.float()).detach()

    
        text_features, image_features = self.graph_learner(image_features)
       

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


@TRAINER_REGISTRY.register()
class GraphCLIP_v1(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]
    
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)

        

        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model, self.train_loader_x).cuda()
        # for key, value in self.model.named_parameters():
        #     print(key, value)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "graph_learner" not in name:
                param.requires_grad_(False)

        for param in self.model.graph_learner.parameters():
            param.requires_grad_(True)


        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.graph_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.model.float()
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(model=self.model.graph_learner, optim_cfg=cfg.OPTIM)
    #    , optim_cfg, param_groups=None
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("graph_learner", self.model.graph_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        
        prec = self.cfg.TRAINER.COOP.PREC
        # print("========prec", prec)
    
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
        

"""Preserve V4 learned SnakeBeta activations, not V3's fixed leaky-ReLU."""
import torch

class SnakeBeta(torch.nn.Module):
    def __init__(self,source):
        super().__init__()
        self.register_buffer("alpha",source.alpha.clone().reshape(1,-1,1))
        self.register_buffer("beta",source.beta.clone().reshape(1,-1,1))
        self.logscale=bool(source.alpha_logscale)
        self.epsilon=float(source.no_div_by_zero)
    def forward(self,x):
        alpha=torch.exp(self.alpha) if self.logscale else self.alpha
        beta=torch.exp(self.beta) if self.logscale else self.beta
        return x+(1.0/(beta+self.epsilon))*torch.sin(x*alpha).square()

def activation(source):
    name=getattr(source,"original_name",type(source).__name__)
    if name=="SnakeBeta":
        return SnakeBeta(source)
    if name=="LeakyReLU":
        return torch.nn.LeakyReLU(float(source.negative_slope))
    raise ValueError("Unsupported original vocoder activation: "+name)

def make_vocoder(source,base):
    if not hasattr(source,"activation_ups"):
        return base.Vocoder(source)
    class Residual(torch.nn.Module):
        def __init__(self,original):
            super().__init__()
            self.convs=torch.nn.ModuleList(base.copy_conv(c) for c in original.convs.children())
            self.acts=torch.nn.ModuleList(activation(a) for a in original.activations.children())
            assert len(self.convs)==len(self.acts)
        def forward(self,x):
            for conv,act in zip(self.convs,self.acts):
                x=x+conv(act(x))
            return x
    class LearnedVocoder(base.Vocoder):
        def __init__(self,original):
            super().__init__(original)
            self.blocks=torch.nn.ModuleList(Residual(c) for c in original.resblocks_u.children())
            self.acts=torch.nn.ModuleList(activation(a) for a in original.activation_ups.children())
            self.post_act=activation(original.activation_post)
            assert len(self.acts)==len(self.ups)
        def forward(self,mel):
            value=self.pre(mel)
            for i,up in enumerate(self.ups):
                value=up(self.acts[i](value))
                total=self.blocks[i*self.kernels](value)
                for j in range(1,self.kernels):
                    total=total+self.blocks[i*self.kernels+j](value)
                value=total/self.kernels
            value=self.post(self.post_act(value))
            value=torch.nn.functional.conv_transpose1d(value,self.updown_filter*self.bands,stride=self.bands)
            value=torch.nn.functional.conv1d(value,self.synthesis_filter,padding=self.pad)
            return torch.where(torch.amax(torch.abs(value))<=1,value,torch.tanh(value))
    return LearnedVocoder(source)

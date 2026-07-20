from types import SimpleNamespace
import torch
from osediff_focus_fusion import expand_unet_conv_in, tiled_unet_forward

class DummyUNet(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.conv_in=torch.nn.Conv2d(4,8,3,padding=1); self.conv_out=torch.nn.Conv2d(8,4,1)
        self.config=SimpleNamespace(in_channels=4,out_channels=4)
    def register_to_config(self, **kw):
        for k,v in kw.items(): setattr(self.config,k,v)
    def forward(self,x,timestep,encoder_hidden_states=None): return SimpleNamespace(sample=self.conv_out(torch.relu(self.conv_in(x))))

def test_expand_preserves_base_and_zeros_extra():
    u=DummyUNet(); weight=u.conv_in.weight.detach().clone(); bias=u.conv_in.bias.detach().clone()
    expand_unet_conv_in(u,10)
    assert u.conv_in.in_channels==u.config.in_channels==10
    torch.testing.assert_close(u.conv_in.weight[:,:4],weight,rtol=0,atol=0)
    assert torch.count_nonzero(u.conv_in.weight[:,4:])==0
    torch.testing.assert_close(u.conv_in.bias,bias,rtol=0,atol=0)
    assert u(torch.randn(2,10,8,8),torch.tensor([999])).sample.shape==(2,4,8,8)

def test_tiled_10_to_4_shape():
    u=expand_unet_conv_in(DummyUNet(),10); x=torch.randn(1,10,19,23)
    y=tiled_unet_forward(u,x,torch.tensor([999]),torch.empty(1,1,1),4,8,3)
    assert y.shape==(1,4,19,23)

def test_scheduler_receives_four_channel_sample():
    class Scheduler:
        def step(self,pred,t,sample,return_dict=True):
            assert pred.shape[1]==sample.shape[1]==4
            return SimpleNamespace(prev_sample=sample-pred)
    u=expand_unet_conv_in(DummyUNet(),10); z_a=torch.randn(1,4,8,8); z_b=torch.randn(1,4,8,8)
    inp=torch.cat([z_a,z_b,torch.rand(1,1,8,8),torch.rand(1,1,8,8)],1)
    pred=u(inp,torch.tensor([999])).sample; out=Scheduler().step(pred,None,z_a).prev_sample
    assert inp.shape[1]==10 and pred.shape[1]==out.shape[1]==4


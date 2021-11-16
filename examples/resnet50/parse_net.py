import torch
from torchvision.models import resnet50

def main():
    import sys
    if len(sys.argv) != 2:
        print('{} <.pth>'.format(sys.argv[0]))
        sys.exit(1)
    model_path = sys.argv[1]
    model = resnet50(pretrained=False)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    dummy_input = torch.rand(1, 3, 224, 224)
    model = torch.jit.trace(model, dummy_input)
    traced_model_path = model_path.replace('.pth', '.traced.pth')
    model.save(traced_model_path)
    import bmnetp
    net_name = 'torch-resnet50-v1'
    bmnetp.compile(
        model=traced_model_path,
        net_name=net_name,
        outdir='{}.fp32compilation'.format(net_name),
        shapes=[[1, 3, 224, 224]],
        cmp=True,
        target='BM1684')
    import ufw.tools as tools
    tools.pt_to_umodel([
        '-m', traced_model_path,
        '-d', '{}.ufwcompilation'.format(net_name),
        '-s', '(1,3,224,224)',
        '-n', net_name,
        ])

if __name__ == '__main__':
    main()


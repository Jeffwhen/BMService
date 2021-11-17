import ufw.tools as tools

def compile_ssd_resnet34(model_path):
    import bmneto
    net_name = 'ssd-resnet34'
    input_name = 'image'
    output_names = ['Concat_659', 'Softmax_660']
    input_shape = [1, 3, 1200, 1200]
    bmneto.compile(
        model=model_path,
        net_name=net_name,
        outdir=f'{net_name}.fp32compilation',
        target='BM1684',
        shapes=[input_shape],
        input_names=[input_name],
        output_names=output_names,
        cmp=True)
    tools.on_to_umodel([
        '-m', model_path,
        '-n', net_name,
        '-i', f'[{input_name}]',
        '-o', f'[{",".join(output_names)}]',
        '-s', f'{",".join(str(i) for i in input_shape)}',
        '-d', f'{net_name}.ufwcompilation',
        '--cmp'])

def compile_ssd_mobilenet(model_path):
    import bmnett
    net_name = 'ssd-mobilenet'
    input_name = 'normalized_input_image_tensor'
    output_names = ['raw_outputs/class_predictions', 'raw_outputs/box_encodings']
    input_shape = [1, 300, 300, 3]
    bmnett.compile(
        model=model_path,
        net_name=net_name,
        outdir=f'{net_name}.fp32compilation',
        target='BM1684',
        shapes=[input_shape],
        input_names=[input_name],
        output_names=output_names,
        cmp=True)
    tools.tf_to_umodel([
        '-m', model_path,
        '-n', net_name,
        '-i', f'[{input_name}]',
        '-o', f'[{",".join(output_names)}]',
        '-s', f'{",".join(str(i) for i in input_shape)}',
        '-d', f'{net_name}.ufwcompilation',
        '--cmp'])

def main():
    import sys
    if len(sys.argv) != 2:
        print(f'{sys.argv[0]} <.onnx>')
        return

    model_path = sys.argv[1]
    if model_path.endswith('.onnx'):
        compile_ssd_resnet34(model_path)
    else:
        compile_ssd_mobilenet(model_path)

if __name__ == '__main__':
    main()

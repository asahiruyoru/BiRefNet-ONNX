import copy
import cv2
import numpy as np
import onnxruntime
import gradio as gr


def load_model(imsize):
    onnx_session = onnxruntime.InferenceSession(
        f'birefnet_{imsize}x{imsize}.onnx',
        providers=[
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ],
    )
    onnx_session.imsize = imsize
    return onnx_session

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def predict(img_path, background_mode="original"):
    """
    background_mode:
        "original" : 透過部分はそのまま（元画像の色）
        "black"    : 透過部分を黒 (0,0,0)
        "white"    : 透過部分を白 (255,255,255)
    """
    imsize = onnx_session.imsize
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_ori = copy.deepcopy(image)
    image_width, image_height = image.shape[1], image.shape[0]

    image = cv2.resize(image, dsize=(imsize, imsize))
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image = (image / 255 - mean) / std
    image = image.transpose(2, 0, 1).astype('float32')
    image = image.reshape(-1, 3, imsize, imsize)

    input_name = onnx_session.get_inputs()[0].name
    results = onnx_session.run(None, {input_name: image})

    mask_image = np.squeeze(results[-1])
    mask_image = sigmoid(mask_image)
    mask_image *= 255
    mask_image = mask_image.astype('uint8')
    mask_image = cv2.resize(mask_image, dsize=(image_width, image_height))
    binary_mask_image = np.where(mask_image > 127, 255, 0).astype("uint8")

    rgba_foreground = np.concatenate([image_ori, binary_mask_image[..., np.newaxis]], axis=-1)

    background_alpha = np.where(mask_image > 127, 0, 255).astype("uint8")
    rgba_background = cv2.cvtColor(image_ori, cv2.COLOR_RGB2RGBA)
    rgba_background[:, :, 3] = background_alpha

    # === ★ 背景色選択ロジック ===
    if background_mode.lower() == "black":
        color = (0, 0, 0)
        rgba_foreground[rgba_foreground[:, :, 3] == 0, :3] = color
        rgba_background[rgba_background[:, :, 3] == 0, :3] = color

    elif background_mode.lower() == "white":
        color = (255, 255, 255)
        rgba_foreground[rgba_foreground[:, :, 3] == 0, :3] = color
        rgba_background[rgba_background[:, :, 3] == 0, :3] = color
    # "original" の場合は何もしない
    # =====================================

    return binary_mask_image, rgba_foreground, rgba_background

class BiRefNet:
    def __init__(self):
        self.demo = self.layout()
        self.input_image_path = None

    def layout(self):
        css = """
        #intro{
            max-width: 32rem;
            text-align: center;
            margin: 0 auto;
        }
        """
        with gr.Blocks(css=css) as demo:
            with gr.Row():
                with gr.Column():
                    self.input_image_path = gr.Image(label="Input image", type='filepath')
                    self.bg_mode = gr.Radio(
                        ["original", "black", "white"],
                        label="背景の扱い",
                        value="black",
                    )
                    generate_button = gr.Button(value="Generate", variant="primary")
                with gr.Column():
                    self.output_image = gr.Image(type="pil", label="Output image", format="png")
                    self.foreground_image = gr.Image(type="pil", label="foreground image", format="png")
                    self.background_image = gr.Image(type="pil", label="background image", format="png")

            generate_button.click(
                fn=predict,
                inputs=[self.input_image_path, self.bg_mode],
                outputs=[self.output_image, self.foreground_image, self.background_image],
            )
        return demo

imsize = 1024
onnx_session = load_model(imsize)

birefnet = BiRefNet()
birefnet.demo.queue()
birefnet.demo.launch(share=False)

if False:
    import argparse
    parser = argparse.ArgumentParser(description='BirefNet')
    parser.add_argument('--imsize', type=int, default=1024, help='input image size')
    parser.add_argument('--img_path', type=str, help='path to input image', default='sample.jpg')
    args = parser.parse_args()

    onnx_session = load_model(args.imsize)
    mask_image = predict(args.img_path, onnx_session)

    cv2.imwrite('mask.png', mask_image)
    print('Mask saved as mask.png')     

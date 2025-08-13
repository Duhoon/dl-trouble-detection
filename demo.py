import gradio as gr
from Classifier import Classifier

clf = Classifier()

def predict_image(image):
    res = clf.classify_pic(image)
    return res

with gr.Blocks() as demo:
    gr.Markdown("# 안면피부진단 서비스")
    gr.Markdown("질환이 의심되는 부위를 이미지로 찍어 업로드 해주세요.")
    
    with gr.Row():
        classifier_interface = gr.Interface(
            fn=predict_image,
            inputs=gr.Image(type='pil'),
            outputs=gr.Text(label="질환 분류")
        )
    
    gr.Markdown("# 질환 조치")
    btn_postprocess = gr.Button("조치 확인하기")
    postprocess = gr.Textbox(lines=15)
    
    # btn_postprocess.click(outputs=txt)

demo.launch()
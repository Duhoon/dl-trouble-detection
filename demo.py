import gradio as gr
from Classifier import Classifier
from Advisor import Advisor

clf = Classifier()
adv = Advisor()

def predict_image(image):
    res = clf.classify_pic(image)
    return res

with gr.Blocks() as demo:
    gr.Markdown("# 안면피부진단 서비스")
    gr.Markdown("질환이 의심되는 부위를 이미지로 찍어 업로드 해주세요.")
    
    with gr.Row():
        img_input = gr.Image(type='pil')
        output_res = gr.Text(label="질환 분류")
    btn = gr.Button("분류하기")
    btn.click(fn=clf.classify_pic, inputs=img_input, outputs=output_res)
    
    gr.Markdown("# 질환 조치")
    btn_postprocess = gr.Button("조치 확인하기")
    postprocess = gr.Textbox(lines=15)
    
    btn_postprocess.click(fn=adv.answer, inputs=output_res, outputs=postprocess)

demo.launch()
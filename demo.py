import gradio as gr
from Classifier import Classifier
from Advisor import Advisor
import pandas as pd

clf = Classifier()
adv = Advisor()

def predict_image(image):
    res = clf.classify_pic(image)
    return res

def interface():
    with gr.Blocks() as demo:
        with gr.Tab("예측기"):
            gr.Markdown("# 안면피부진단 서비스")
            gr.Markdown("질환이 의심되는 부위를 이미지로 찍어 업로드 해주세요.")
            model_name = gr.Dropdown(clf.model.keys(), label="모델 선택")

            with gr.Row():
                img_input = gr.Image(type='pil')
                output_res = gr.Text(label="질환 분류")
                output_barplot = gr.BarPlot(pd.DataFrame({"질환": ["없음"], "확률": [0]}), x="질환", y="확률")
                
            btn = gr.Button("분류하기")
            btn.click(fn=clf.classify_pic, inputs=[img_input, model_name], outputs=[output_res,output_barplot])
            
            gr.Markdown("# 질환 조치")
            btn_postprocess = gr.Button("조치 확인하기")
            postprocess = gr.Textbox(lines=15)
            btn_postprocess.click(fn=adv.answer, inputs=output_res, outputs=postprocess)

        with gr.Tab("챗봇"):
            gr.Chatbot()
    
    return demo

if __name__ == "__main__":
    demo = interface()
    demo.launch(server_port=7861, server_name="0.0.0.0", debug=True)
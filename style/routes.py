from msilib.schema import File
from flask import Flask, render_template, request, redirect,send_file,  flash, abort, url_for
from style import app,db,mail
from style import app,db,mail
from style import app
from style.models import *
# from flask_login import login_user, current_user, logout_user, login_required

from random import randint
import os
from PIL import Image
from flask_mail import Message
from io import BytesIO
import glob
import time
import cv2
import argparse
from style_transfer.inference import inference
from style_transfer.transform_net import ImageTransformNet
from utils.parse_arguments import get_style_webcam_arguments
UPLOAD_FOLDER= "C:/Users/HP/OneDrive/Desktop/STYLE_CONVERTER/data/images"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



UPLOAD_FOLDER1= "C:/Users/HP/OneDrive/Desktop/STYLE_CONVERTER/data/images"
app.config['UPLOAD_FOLDER1'] = UPLOAD_FOLDER1




UPLOAD_FOLDER2= "C:/Users/HP/OneDrive/Desktop/STYLE_CONVERTER/data/videos"
app.config['UPLOAD_FOLDER2'] = UPLOAD_FOLDER2

@app.route('/')
def index1():
    return render_template("index1.html")




@app.route('/det',methods=['GET','POST'])
def det():
    import os
    if request.method == "POST":

        image = request.files['image']
        filename = "content.png"
        image.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
        #style = request.files['style']
        #filename1 = "style.png"
        #style.save(os.path.join(app.config['UPLOAD_FOLDER1'],filename1))
        from style_transfer.inference import inference
        from style_transfer.transform_net import ImageTransformNet
        from utils.utility import load_image, load_url_image, array_to_img
        from utils.parse_arguments import get_style_image_arguments
        import argparse
        import time

        # parser = argparse.ArgumentParser(description="Create Styled Images")

        # parser.add_argument("--config", "-con", help="Path to style image config file")
        # parser.add_argument("--checkpoint", "-ckpt", help="Path to trained style checkpoints")
        # parser.add_argument("--image", "-img", help="url or file path to image to style")
        # parser.add_argument(
        #     "--image_size",
        #     "-size",
        #     nargs="+",
        #     type=int,
        #     default=[256, 256],
        #     help="output image size",
        # )
        # parser.add_argument(
        #     "--output", "-out", default="output/styled.jpg", help="Path to save output image"
        # )

        # args = get_style_image_arguments(parser.parse_args())


        style_model = ImageTransformNet()
        style_model.load_weights('C:/Users/HP/OneDrive/Desktop/STYLE_CONVERTER/data/models/udnie/model_checkpoint.ckpt')
        input_shape = [256, 256] 
        image_path = 'data/images/content.png'
        if image_path.startswith("http"):
            image = load_url_image(image_path, dim=input_shape)
        else:
            image = load_image(image_path, dim=input_shape)

        start = time.time()
        styled_image = inference(style_model, image)
        end = time.time()
        print(f"Time Taken: {end-start:.2f}s")
        pil_image = array_to_img(styled_image)
        pil_image.show()
        pil_image.save('output/styled.jpg')
        filelist=glob.glob("C:/Users/HP/OneDrive/Desktop/STYLE_CONVERTER/data/images/*.*")
        for filepath in filelist:
            try:
                os.remove(filepath)
            except:
                print("Error while deleting file")

     
    
    
    return render_template("up_image1.html")








@app.route('/det1',methods=['GET','POST'])
def det1():
    import os
    if request.method == "POST":

        image = request.files['style']
        filename = "style.png"
        image.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
        video = request.files['video']
        filename1 = "test.mp4"
        video.save(os.path.join(app.config['UPLOAD_FOLDER2'],filename1))
        import time
        import cv2
        import argparse
        from style_transfer.inference import inference
        from style_transfer.transform_net import ImageTransformNet
        from utils.parse_arguments import get_style_video_arguments


        # parser = argparse.ArgumentParser(description="Create Stylized Videos")
        # parser.add_argument("--config", "-con", help="Path to style image config file")
        # parser.add_argument("--checkpoint", "-ckpt", help="Trained Checkpoints Path")
        # parser.add_argument("--output", "-out", help="Output Path for saving recorded video")
        # parser.add_argument(
        #     "--format",
        #     "-format",
        #     default="XVID",
        #     help="codec used in VideoWriter when saving video to file",
        # )
        # parser.add_argument("--video", "-vid", help="Path to video file to process")
        # parser.add_argument("--size", "-s", nargs="+", type=int, help="output video size")
        # args = get_style_video_arguments(parser.parse_args())


        style_model = ImageTransformNet()
        style_model.load_weights('C:/Users/HP/OneDrive/Desktop/STYLE_CONVERTER/data/models/udnie/model_checkpoint.ckpt')


        vid = cv2.VideoCapture('data/videos/test.mp4')
        out = None
        output="output/styled_video.avi"
        size=[640,480]
        if output:
            width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if size:
                width = int(size[0])
                height = int(size[1])
            fps = int(vid.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output, codec, fps, (width, height))

        start = time.time()
        while True:
            _, img = vid.read()

            if img is None:
                print("Empty Frame")
                time.sleep(0.1)
                break
            img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if size:
                img_in = cv2.resize(img_in, tuple(size))
            styled_image = inference(style_model, img_in)
            if out:
                out.write(styled_image)
            cv2.imshow("output", styled_image)
            if cv2.waitKey(1) == ord("q"):
                break
        cv2.destroyAllWindows()
        end = time.time()
        print(f"Time taken: {end-start:.2f}sec")
        filelist=glob.glob("C:/Users/HP/OneDrive/Desktop/STYLE_CONVERTER/data/images/*.*")
        for filepath in filelist:
            try:
                os.remove(filepath)
            except:
                print("Error while deleting file")
        filelist1=glob.glob("C:/Users/HP/OneDrive/Desktop/STYLE_CONVERTER/data/videos/*.*")
        for filepath in filelist1:
            try:
                os.remove(filepath)
            except:
                print("Error while deleting file")

        

     
    
    
    return render_template("up_video.html")







@app.route('/real',methods=['GET','POST'])
def real():
    import os
    if request.method == "POST":

        image = request.files['style']
        filename = "style.png"
        image.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))


        # parser = argparse.ArgumentParser(description="Live Stylized Videos")
        # parser.add_argument("--config", "-con", help="Path to style image config file")
        # parser.add_argument("--checkpoint", "-ckpt", help="Trained Checkpoints Path")
        # parser.add_argument("--output", "-out", help="Output Path for saving recorded video")
        # parser.add_argument(
        #     "--format",
        #     "-format",
        #     default="XVID",
        #     help="codec used in VideoWriter when saving video to file",
        # )
        # parser.add_argument("--size", "-s", nargs="+", type=int, help="output video size")
        # parser.add_argument(
        #     "--camera", "-cam", default=0, type=int, help="webcam number to record"
        # )
        # args = get_style_webcam_arguments(parser.parse_args())


        style_model = ImageTransformNet()
        style_model.load_weights('C:/Users/HP/OneDrive/Desktop/STYLE_CONVERTER/data/models/starry_nights/model_checkpoint.ckpt')
        camera=0
        vid = cv2.VideoCapture(camera)
        out = None
        output='output/webcam.avi'
        size=[640,480]
        if output:
            width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if size:
                width = int(size[0])
                height = int(size[1])
            fps = int(vid.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output, codec, fps, (width, height))

        while True:
            _, img = vid.read()

            if img is None:
                print("Empty Frame")
                time.sleep(0.1)
                break
            img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if size:
                img_in = cv2.resize(img_in, tuple(size))
            styled_image = inference(style_model, img_in)
            if out:
                out.write(styled_image)
            cv2.imshow("output", styled_image)
            if cv2.waitKey(1) == ord("q"):
                break
        cv2.destroyAllWindows()
        filelist=glob.glob("C:/Users/HP/OneDrive/Desktop/STYLE_CONVERTER/data/images/*.*")
        for filepath in filelist:
            try:
                os.remove(filepath)
            except:
                print("Error while deleting file")
    return render_template("real.html")


#image styling

@app.route('/fn',methods=['GET','POST'])
def fn():
    import os
    if request.method == "POST":

        image = request.files['image']
        filename = "content.png"
        image.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
        #style = request.files['style']
        #filename1 = "style.png"
        #style.save(os.path.join(app.config['UPLOAD_FOLDER1'],filename1))
        from style_transfer.inference import inference
        from style_transfer.transform_net import ImageTransformNet
        from utils.utility import load_image, load_url_image, array_to_img
        from utils.parse_arguments import get_style_image_arguments
        import argparse
        import time

        # parser = argparse.ArgumentParser(description="Create Styled Images")

        # parser.add_argument("--config", "-con", help="Path to style image config file")
        # parser.add_argument("--checkpoint", "-ckpt", help="Path to trained style checkpoints")
        # parser.add_argument("--image", "-img", help="url or file path to image to style")
        # parser.add_argument(
        #     "--image_size",
        #     "-size",
        #     nargs="+",
        #     type=int,
        #     default=[256, 256],
        #     help="output image size",
        # )
        # parser.add_argument(
        #     "--output", "-out", default="output/styled.jpg", help="Path to save output image"
        # )

        # args = get_style_image_arguments(parser.parse_args())


        style_model = ImageTransformNet()
        style_model.load_weights('C:/Users/HP/OneDrive/Desktop/STYLE_CONVERTER/data/models/udnie/model_checkpoint.ckpt')
        input_shape = [256, 256] 
        image_path = 'data/images/content.png'
        if image_path.startswith("http"):
            image = load_url_image(image_path, dim=input_shape)
        else:
            image = load_image(image_path, dim=input_shape)

        start = time.time()
        styled_image = inference(style_model, image)
        end = time.time()
        print(f"Time Taken: {end-start:.2f}s")
        pil_image = array_to_img(styled_image)
        pil_image.show()
        pil_image.save('output/styled.jpg')
        filelist=glob.glob("C:/Users/HP/OneDrive/Desktop/STYLE_CONVERTER/data/images/*.*")
        for filepath in filelist:
            try:
                os.remove(filepath)
            except:
                print("Error while deleting file")

     
    
    
    return render_template("udnie.html")







@app.route('/fn1',methods=['GET','POST'])
def fn1():
    import os
    if request.method == "POST":

        image = request.files['image']
        filename = "content.png"
        image.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
        #style = request.files['style']
        #filename1 = "style.png"
        #style.save(os.path.join(app.config['UPLOAD_FOLDER1'],filename1))
        from style_transfer.inference import inference
        from style_transfer.transform_net import ImageTransformNet
        from utils.utility import load_image, load_url_image, array_to_img
        from utils.parse_arguments import get_style_image_arguments
        import argparse
        import time

        # parser = argparse.ArgumentParser(description="Create Styled Images")

        # parser.add_argument("--config", "-con", help="Path to style image config file")
        # parser.add_argument("--checkpoint", "-ckpt", help="Path to trained style checkpoints")
        # parser.add_argument("--image", "-img", help="url or file path to image to style")
        # parser.add_argument(
        #     "--image_size",
        #     "-size",
        #     nargs="+",
        #     type=int,
        #     default=[256, 256],
        #     help="output image size",
        # )
        # parser.add_argument(
        #     "--output", "-out", default="output/styled.jpg", help="Path to save output image"
        # )

        # args = get_style_image_arguments(parser.parse_args())


        style_model = ImageTransformNet()
        style_model.load_weights('C:/Users/HP/OneDrive/Desktop/STYLE_CONVERTER/data/models/scream/model_checkpoint.ckpt')
        input_shape = [256, 256] 
        image_path = 'data/images/content.png'
        if image_path.startswith("http"):
            image = load_url_image(image_path, dim=input_shape)
        else:
            image = load_image(image_path, dim=input_shape)

        start = time.time()
        styled_image = inference(style_model, image)
        end = time.time()
        print(f"Time Taken: {end-start:.2f}s")
        pil_image = array_to_img(styled_image)
        pil_image.show()
        pil_image.save('output/styled.jpg')
        filelist=glob.glob("C:/Users/HP/OneDrive/Desktop/STYLE_CONVERTER/data/images/*.*")
        for filepath in filelist:
            try:
                os.remove(filepath)
            except:
                print("Error while deleting file")

     
    
    
    return render_template("scream.html")




@app.route('/fn2',methods=['GET','POST'])
def fn2():
    import os
    if request.method == "POST":

        image = request.files['image']
        filename = "content.png"
        image.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
        #style = request.files['style']
        #filename1 = "style.png"
        #style.save(os.path.join(app.config['UPLOAD_FOLDER1'],filename1))
        from style_transfer.inference import inference
        from style_transfer.transform_net import ImageTransformNet
        from utils.utility import load_image, load_url_image, array_to_img
        from utils.parse_arguments import get_style_image_arguments
        import argparse
        import time

        # parser = argparse.ArgumentParser(description="Create Styled Images")

        # parser.add_argument("--config", "-con", help="Path to style image config file")
        # parser.add_argument("--checkpoint", "-ckpt", help="Path to trained style checkpoints")
        # parser.add_argument("--image", "-img", help="url or file path to image to style")
        # parser.add_argument(
        #     "--image_size",
        #     "-size",
        #     nargs="+",
        #     type=int,
        #     default=[256, 256],
        #     help="output image size",
        # )
        # parser.add_argument(
        #     "--output", "-out", default="output/styled.jpg", help="Path to save output image"
        # )

        # args = get_style_image_arguments(parser.parse_args())


        style_model = ImageTransformNet()
        style_model.load_weights('C:/Users/HP/OneDrive/Desktop/STYLE_CONVERTER/data/models/starry_nights/model_checkpoint.ckpt')
        input_shape = [256, 256] 
        image_path = 'data/images/content.png'
        if image_path.startswith("http"):
            image = load_url_image(image_path, dim=input_shape)
        else:
            image = load_image(image_path, dim=input_shape)

        start = time.time()
        styled_image = inference(style_model, image)
        end = time.time()
        print(f"Time Taken: {end-start:.2f}s")
        pil_image = array_to_img(styled_image)
        pil_image.show()
        pil_image.save('output/styled.jpg')
        filelist=glob.glob("C:/Users/HP/OneDrive/Desktop/STYLE_CONVERTER/data/images/*.*")
        for filepath in filelist:
            try:
                os.remove(filepath)
            except:
                print("Error while deleting file")

     
    
    return render_template("starry_nights.html")





@app.route('/fn3',methods=['GET','POST'])
def fn3():
    import os
    if request.method == "POST":

        image = request.files['image']
        filename = "content.png"
        image.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
        #style = request.files['style']
        #filename1 = "style.png"
        #style.save(os.path.join(app.config['UPLOAD_FOLDER1'],filename1))
        from style_transfer.inference import inference
        from style_transfer.transform_net import ImageTransformNet
        from utils.utility import load_image, load_url_image, array_to_img
        from utils.parse_arguments import get_style_image_arguments
        import argparse
        import time

        # parser = argparse.ArgumentParser(description="Create Styled Images")

        # parser.add_argument("--config", "-con", help="Path to style image config file")
        # parser.add_argument("--checkpoint", "-ckpt", help="Path to trained style checkpoints")
        # parser.add_argument("--image", "-img", help="url or file path to image to style")
        # parser.add_argument(
        #     "--image_size",
        #     "-size",
        #     nargs="+",
        #     type=int,
        #     default=[256, 256],
        #     help="output image size",
        # )
        # parser.add_argument(
        #     "--output", "-out", default="output/styled.jpg", help="Path to save output image"
        # )

        # args = get_style_image_arguments(parser.parse_args())


        style_model = ImageTransformNet()
        style_model.load_weights('C:/Users/HP/OneDrive/Desktop/STYLE_CONVERTER/data/models/candy/model_checkpoint.ckpt')
        input_shape = [256, 256] 
        image_path = 'data/images/content.png'
        if image_path.startswith("http"):
            image = load_url_image(image_path, dim=input_shape)
        else:
            image = load_image(image_path, dim=input_shape)

        start = time.time()
        styled_image = inference(style_model, image)
        end = time.time()
        print(f"Time Taken: {end-start:.2f}s")
        pil_image = array_to_img(styled_image)
        pil_image.show()
        pil_image.save('output/styled.jpg')
        filelist=glob.glob("C:/Users/HP/OneDrive/Desktop/STYLE_CONVERTER/data/images/*.*")
        for filepath in filelist:
            try:
                os.remove(filepath)
            except:
                print("Error while deleting file")

     
    
    return render_template("candy.html")




   #video styling 


@app.route('/vn',methods=['GET','POST'])
def vn():
    import os
    if request.method == "POST":

        #image = request.files['style']
        #filename = "style.png"
        #image.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
        video = request.files['video']
        filename1 = "test.mp4"
        video.save(os.path.join(app.config['UPLOAD_FOLDER2'],filename1))
        import time
        import cv2
        import argparse
        from style_transfer.inference import inference
        from style_transfer.transform_net import ImageTransformNet
        from utils.parse_arguments import get_style_video_arguments


        # parser = argparse.ArgumentParser(description="Create Stylized Videos")
        # parser.add_argument("--config", "-con", help="Path to style image config file")
        # parser.add_argument("--checkpoint", "-ckpt", help="Trained Checkpoints Path")
        # parser.add_argument("--output", "-out", help="Output Path for saving recorded video")
        # parser.add_argument(
        #     "--format",
        #     "-format",
        #     default="XVID",
        #     help="codec used in VideoWriter when saving video to file",
        # )
        # parser.add_argument("--video", "-vid", help="Path to video file to process")
        # parser.add_argument("--size", "-s", nargs="+", type=int, help="output video size")
        # args = get_style_video_arguments(parser.parse_args())


        style_model = ImageTransformNet()
        style_model.load_weights('C:/Users/HP/OneDrive/Desktop/STYLE_CONVERTER/data/models/udnie/model_checkpoint.ckpt')


        vid = cv2.VideoCapture('data/videos/test.mp4')
        out = None
        output="output/styled_video.avi"
        size=[640,480]
        if output:
            width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if size:
                width = int(size[0])
                height = int(size[1])
            fps = int(vid.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output, codec, fps, (width, height))

        start = time.time()
        while True:
            _, img = vid.read()

            if img is None:
                print("Empty Frame")
                time.sleep(0.1)
                break
            img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if size:
                img_in = cv2.resize(img_in, tuple(size))
            styled_image = inference(style_model, img_in)
            if out:
                out.write(styled_image)
            cv2.imshow("output", styled_image)
            if cv2.waitKey(1) == ord("q"):
                break
        cv2.destroyAllWindows()
        end = time.time()
        print(f"Time taken: {end-start:.2f}sec")
        filelist=glob.glob("C:/Users/HP/OneDrive/Desktop/STYLE_CONVERTER/data/images/*.*")
        for filepath in filelist:
            try:
                os.remove(filepath)
            except:
                print("Error while deleting file")
        filelist1=glob.glob("C:/Users/HP/OneDrive/Desktop/STYLE_CONVERTER/data/videos/*.*")
        for filepath in filelist1:
            try:
                os.remove(filepath)
            except:
                print("Error while deleting file")

        

     
    
    
    return render_template("udnie_video.html")   



    




@app.route('/vn1',methods=['GET','POST'])
def vn1():
    import os
    if request.method == "POST":

        #image = request.files['style']
        #filename = "style.png"
        #image.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
        video = request.files['video']
        filename1 = "test.mp4"
        video.save(os.path.join(app.config['UPLOAD_FOLDER2'],filename1))
        import time
        import cv2
        import argparse
        from style_transfer.inference import inference
        from style_transfer.transform_net import ImageTransformNet
        from utils.parse_arguments import get_style_video_arguments


        # parser = argparse.ArgumentParser(description="Create Stylized Videos")
        # parser.add_argument("--config", "-con", help="Path to style image config file")
        # parser.add_argument("--checkpoint", "-ckpt", help="Trained Checkpoints Path")
        # parser.add_argument("--output", "-out", help="Output Path for saving recorded video")
        # parser.add_argument(
        #     "--format",
        #     "-format",
        #     default="XVID",
        #     help="codec used in VideoWriter when saving video to file",
        # )
        # parser.add_argument("--video", "-vid", help="Path to video file to process")
        # parser.add_argument("--size", "-s", nargs="+", type=int, help="output video size")
        # args = get_style_video_arguments(parser.parse_args())


        style_model = ImageTransformNet()
        style_model.load_weights('C:/Users/HP/OneDrive/Desktop/STYLE_CONVERTER/data/models/scream/model_checkpoint.ckpt')


        vid = cv2.VideoCapture('data/videos/test.mp4')
        out = None
        output="output/styled_video.avi"
        size=[640,480]
        if output:
            width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if size:
                width = int(size[0])
                height = int(size[1])
            fps = int(vid.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output, codec, fps, (width, height))

        start = time.time()
        while True:
            _, img = vid.read()

            if img is None:
                print("Empty Frame")
                time.sleep(0.1)
                break
            img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if size:
                img_in = cv2.resize(img_in, tuple(size))
            styled_image = inference(style_model, img_in)
            if out:
                out.write(styled_image)
            cv2.imshow("output", styled_image)
            if cv2.waitKey(1) == ord("q"):
                break
        cv2.destroyAllWindows()
        end = time.time()
        print(f"Time taken: {end-start:.2f}sec")
        filelist=glob.glob("C:/Users/HP/OneDrive/Desktop/STYLE_CONVERTER/data/images/*.*")
        for filepath in filelist:
            try:
                os.remove(filepath)
            except:
                print("Error while deleting file")
        filelist1=glob.glob("C:/Users/HP/OneDrive/Desktop/STYLE_CONVERTER/data/videos/*.*")
        for filepath in filelist1:
            try:
                os.remove(filepath)
            except:
                print("Error while deleting file")

        

     
    
    
    return render_template("scream_video.html")





@app.route('/vn2',methods=['GET','POST'])
def vn2():
    import os
    if request.method == "POST":

        #image = request.files['style']
        #filename = "style.png"
        #image.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
        video = request.files['video']
        filename1 = "test.mp4"
        video.save(os.path.join(app.config['UPLOAD_FOLDER2'],filename1))
        import time
        import cv2
        import argparse
        from style_transfer.inference import inference
        from style_transfer.transform_net import ImageTransformNet
        from utils.parse_arguments import get_style_video_arguments


        # parser = argparse.ArgumentParser(description="Create Stylized Videos")
        # parser.add_argument("--config", "-con", help="Path to style image config file")
        # parser.add_argument("--checkpoint", "-ckpt", help="Trained Checkpoints Path")
        # parser.add_argument("--output", "-out", help="Output Path for saving recorded video")
        # parser.add_argument(
        #     "--format",
        #     "-format",
        #     default="XVID",
        #     help="codec used in VideoWriter when saving video to file",
        # )
        # parser.add_argument("--video", "-vid", help="Path to video file to process")
        # parser.add_argument("--size", "-s", nargs="+", type=int, help="output video size")
        # args = get_style_video_arguments(parser.parse_args())


        style_model = ImageTransformNet()
        style_model.load_weights('C:/Users/HP/OneDrive/Desktop/STYLE_CONVERTER/data/models/starry_nights/model_checkpoint.ckpt')


        vid = cv2.VideoCapture('data/videos/test.mp4')
        out = None
        output="output/styled_video.avi"
        size=[640,480]
        if output:
            width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if size:
                width = int(size[0])
                height = int(size[1])
            fps = int(vid.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output, codec, fps, (width, height))

        start = time.time()
        while True:
            _, img = vid.read()

            if img is None:
                print("Empty Frame")
                time.sleep(0.1)
                break
            img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if size:
                img_in = cv2.resize(img_in, tuple(size))
            styled_image = inference(style_model, img_in)
            if out:
                out.write(styled_image)
            cv2.imshow("output", styled_image)
            if cv2.waitKey(1) == ord("q"):
                break
        cv2.destroyAllWindows()
        end = time.time()
        print(f"Time taken: {end-start:.2f}sec")
        filelist=glob.glob("C:/Users/HP/OneDrive/Desktop/STYLE_CONVERTER/data/images/*.*")
        for filepath in filelist:
            try:
                os.remove(filepath)
            except:
                print("Error while deleting file")
        filelist1=glob.glob("C:/Users/HP/OneDrive/Desktop/STYLE_CONVERTER/data/videos/*.*")
        for filepath in filelist1:
            try:
                os.remove(filepath)
            except:
                print("Error while deleting file")

        

     
    
    
    return render_template("starry_video.html")




@app.route('/vn3',methods=['GET','POST'])
def vn3():
    import os
    if request.method == "POST":

        #image = request.files['style']
        #filename = "style.png"
        #image.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
        video = request.files['video']
        filename1 = "test.mp4"
        video.save(os.path.join(app.config['UPLOAD_FOLDER2'],filename1))
        import time
        import cv2
        import argparse
        from style_transfer.inference import inference
        from style_transfer.transform_net import ImageTransformNet
        from utils.parse_arguments import get_style_video_arguments


        # parser = argparse.ArgumentParser(description="Create Stylized Videos")
        # parser.add_argument("--config", "-con", help="Path to style image config file")
        # parser.add_argument("--checkpoint", "-ckpt", help="Trained Checkpoints Path")
        # parser.add_argument("--output", "-out", help="Output Path for saving recorded video")
        # parser.add_argument(
        #     "--format",
        #     "-format",
        #     default="XVID",
        #     help="codec used in VideoWriter when saving video to file",
        # )
        # parser.add_argument("--video", "-vid", help="Path to video file to process")
        # parser.add_argument("--size", "-s", nargs="+", type=int, help="output video size")
        # args = get_style_video_arguments(parser.parse_args())


        style_model = ImageTransformNet()
        style_model.load_weights('C:/Users/HP/OneDrive/Desktop/STYLE_CONVERTER/data/models/candy/model_checkpoint.ckpt')


        vid = cv2.VideoCapture('data/videos/test.mp4')
        out = None
        output="output/styled_video.avi"
        size=[640,480]
        if output:
            width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if size:
                width = int(size[0])
                height = int(size[1])
            fps = int(vid.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output, codec, fps, (width, height))

        start = time.time()
        while True:
            _, img = vid.read()

            if img is None:
                print("Empty Frame")
                time.sleep(0.1)
                break
            img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if size:
                img_in = cv2.resize(img_in, tuple(size))
            styled_image = inference(style_model, img_in)
            if out:
                out.write(styled_image)
            cv2.imshow("output", styled_image)
            if cv2.waitKey(1) == ord("q"):
                break
        cv2.destroyAllWindows()
        end = time.time()
        print(f"Time taken: {end-start:.2f}sec")
        filelist=glob.glob("C:/Users/HP/OneDrive/Desktop/STYLE_CONVERTER/data/images/*.*")
        for filepath in filelist:
            try:
                os.remove(filepath)
            except:
                print("Error while deleting file")
        filelist1=glob.glob("C:/Users/HP/OneDrive/Desktop/STYLE_CONVERTER/data/videos/*.*")
        for filepath in filelist1:
            try:
                os.remove(filepath)
            except:
                print("Error while deleting file")

        

     
    
    
    return render_template("candy_video.html")



#webcamera



@app.route('/rn',methods=['GET','POST'])
def rn():
    import os
    if request.method == "POST":

        #image = request.files['style']
        #filename = "style.png"
        #image.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))


        # parser = argparse.ArgumentParser(description="Live Stylized Videos")
        # parser.add_argument("--config", "-con", help="Path to style image config file")
        # parser.add_argument("--checkpoint", "-ckpt", help="Trained Checkpoints Path")
        # parser.add_argument("--output", "-out", help="Output Path for saving recorded video")
        # parser.add_argument(
        #     "--format",
        #     "-format",
        #     default="XVID",
        #     help="codec used in VideoWriter when saving video to file",
        # )
        # parser.add_argument("--size", "-s", nargs="+", type=int, help="output video size")
        # parser.add_argument(
        #     "--camera", "-cam", default=0, type=int, help="webcam number to record"
        # )
        # args = get_style_webcam_arguments(parser.parse_args())


        style_model = ImageTransformNet()
        style_model.load_weights('C:/Users/HP/OneDrive/Desktop/STYLE_CONVERTER/data/models/udnie/model_checkpoint.ckpt')
        camera=0
        vid = cv2.VideoCapture(camera)
        out = None
        output='output/webcam.avi'
        size=[640,480]
        if output:
            width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if size:
                width = int(size[0])
                height = int(size[1])
            fps = int(vid.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output, codec, fps, (width, height))

        while True:
            _, img = vid.read()

            if img is None:
                print("Empty Frame")
                time.sleep(0.1)
                break
            img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if size:
                img_in = cv2.resize(img_in, tuple(size))
            styled_image = inference(style_model, img_in)
            if out:
                out.write(styled_image)
            cv2.imshow("output", styled_image)
            if cv2.waitKey(1) == ord("q"):
                break
        cv2.destroyAllWindows()
        filelist=glob.glob("C:/Users/HP/OneDrive/Desktop/STYLE_CONVERTER/data/images/*.*")
        for filepath in filelist:
            try:
                os.remove(filepath)
            except:
                print("Error while deleting file")
    return render_template("udnie_real.html")




@app.route('/rn1',methods=['GET','POST'])
def rn1():
    import os
    if request.method == "POST":

        #image = request.files['style']
        #filename = "style.png"
        #image.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))


        # parser = argparse.ArgumentParser(description="Live Stylized Videos")
        # parser.add_argument("--config", "-con", help="Path to style image config file")
        # parser.add_argument("--checkpoint", "-ckpt", help="Trained Checkpoints Path")
        # parser.add_argument("--output", "-out", help="Output Path for saving recorded video")
        # parser.add_argument(
        #     "--format",
        #     "-format",
        #     default="XVID",
        #     help="codec used in VideoWriter when saving video to file",
        # )
        # parser.add_argument("--size", "-s", nargs="+", type=int, help="output video size")
        # parser.add_argument(
        #     "--camera", "-cam", default=0, type=int, help="webcam number to record"
        # )
        # args = get_style_webcam_arguments(parser.parse_args())


        style_model = ImageTransformNet()
        style_model.load_weights('C:/Users/HP/OneDrive/Desktop/STYLE_CONVERTER/data/models/scream/model_checkpoint.ckpt')
        camera=0
        vid = cv2.VideoCapture(camera)
        out = None
        output='output/webcam.avi'
        size=[640,480]
        if output:
            width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if size:
                width = int(size[0])
                height = int(size[1])
            fps = int(vid.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output, codec, fps, (width, height))

        while True:
            _, img = vid.read()

            if img is None:
                print("Empty Frame")
                time.sleep(0.1)
                break
            img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if size:
                img_in = cv2.resize(img_in, tuple(size))
            styled_image = inference(style_model, img_in)
            if out:
                out.write(styled_image)
            cv2.imshow("output", styled_image)
            if cv2.waitKey(1) == ord("q"):
                break
        cv2.destroyAllWindows()
        filelist=glob.glob("C:/Users/HP/OneDrive/Desktop/STYLE_CONVERTER/data/images/*.*")
        for filepath in filelist:
            try:
                os.remove(filepath)
            except:
                print("Error while deleting file")
    return render_template("scream_real.html")





@app.route('/rn2',methods=['GET','POST'])
def rn2():
    import os
    if request.method == "POST":

        #image = request.files['style']
        #filename = "style.png"
        #image.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))


        # parser = argparse.ArgumentParser(description="Live Stylized Videos")
        # parser.add_argument("--config", "-con", help="Path to style image config file")
        # parser.add_argument("--checkpoint", "-ckpt", help="Trained Checkpoints Path")
        # parser.add_argument("--output", "-out", help="Output Path for saving recorded video")
        # parser.add_argument(
        #     "--format",
        #     "-format",
        #     default="XVID",
        #     help="codec used in VideoWriter when saving video to file",
        # )
        # parser.add_argument("--size", "-s", nargs="+", type=int, help="output video size")
        # parser.add_argument(
        #     "--camera", "-cam", default=0, type=int, help="webcam number to record"
        # )
        # args = get_style_webcam_arguments(parser.parse_args())


        style_model = ImageTransformNet()
        style_model.load_weights('C:/Users/HP/OneDrive/Desktop/STYLE_CONVERTER/data/models/starry_nights/model_checkpoint.ckpt')
        camera=0
        vid = cv2.VideoCapture(camera)
        out = None
        output='output/webcam.avi'
        size=[640,480]
        if output:
            width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if size:
                width = int(size[0])
                height = int(size[1])
            fps = int(vid.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output, codec, fps, (width, height))

        while True:
            _, img = vid.read()

            if img is None:
                print("Empty Frame")
                time.sleep(0.1)
                break
            img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if size:
                img_in = cv2.resize(img_in, tuple(size))
            styled_image = inference(style_model, img_in)
            if out:
                out.write(styled_image)
            cv2.imshow("output", styled_image)
            if cv2.waitKey(1) == ord("q"):
                break
        cv2.destroyAllWindows()
        filelist=glob.glob("C:/Users/HP/OneDrive/Desktop/STYLE_CONVERTER/data/images/*.*")
        for filepath in filelist:
            try:
                os.remove(filepath)
            except:
                print("Error while deleting file")
    return render_template("starry_real.html")





@app.route('/rn3',methods=['GET','POST'])
def rn3():
    import os
    if request.method == "POST":

        #image = request.files['style']
        #filename = "style.png"
        #image.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))


        # parser = argparse.ArgumentParser(description="Live Stylized Videos")
        # parser.add_argument("--config", "-con", help="Path to style image config file")
        # parser.add_argument("--checkpoint", "-ckpt", help="Trained Checkpoints Path")
        # parser.add_argument("--output", "-out", help="Output Path for saving recorded video")
        # parser.add_argument(
        #     "--format",
        #     "-format",
        #     default="XVID",
        #     help="codec used in VideoWriter when saving video to file",
        # )
        # parser.add_argument("--size", "-s", nargs="+", type=int, help="output video size")
        # parser.add_argument(
        #     "--camera", "-cam", default=0, type=int, help="webcam number to record"
        # )
        # args = get_style_webcam_arguments(parser.parse_args())


        style_model = ImageTransformNet()
        style_model.load_weights('C:/Users/HP/OneDrive/Desktop/STYLE_CONVERTER/data/models/candy/model_checkpoint.ckpt')
        camera=0
        vid = cv2.VideoCapture(camera)
        out = None
        output='output/webcam.avi'
        size=[640,480]
        if output:
            width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if size:
                width = int(size[0])
                height = int(size[1])
            fps = int(vid.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output, codec, fps, (width, height))

        while True:
            _, img = vid.read()

            if img is None:
                print("Empty Frame")
                time.sleep(0.1)
                break
            img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if size:
                img_in = cv2.resize(img_in, tuple(size))
            styled_image = inference(style_model, img_in)
            if out:
                out.write(styled_image)
            cv2.imshow("output", styled_image)
            if cv2.waitKey(1) == ord("q"):
                break
        cv2.destroyAllWindows()
        filelist=glob.glob("C:/Users/HP/OneDrive/Desktop/STYLE_CONVERTER/data/images/*.*")
        for filepath in filelist:
            try:
                os.remove(filepath)
            except:
                print("Error while deleting file")
    return render_template("candy_real.html")

from . import ml5_nn
from . import utilis
import jp_proxy_widget
from IPython.display import display
from jupyter_ui_poll import ui_events
import numpy as np
import time


class ObjectDetector(ml5_nn.neuralNetwork):

    def __init__(self, model, options=None, *pargs, **kwargs):
        super(ObjectDetector,self).__init__(options=options,*pargs, **kwargs)
        self.data = []
        if options is None:
            options = self.default_options()
        self.element.html("Loaded ml5.js")
        self.detect_result = []
        self.count = 0
        self.detect = False
        self.model_load = False
        def model_ready():
            self.model_load = True

        self.js_init("""
            element.nn_info = {};
            const model = ml5.objectDetector(model_name, options = options, callback = modelReady);
            element.nn_info.network = model;
            function modelReady() {
                console.log('Model Ready!');
                model_ready()
            }
            element.predict_images = [];
        """,model_name = model, model_ready=model_ready, options = self.options)
        with ui_events() as poll:
            while self.model_load is False:
                poll(10)
                print('.', end='')
                time.sleep(0.1)
        print('Modeal is ready')

    def default_options(self):
        return {'filterBoxesThreshold': 0.01, 
                'IOUThreshold': 0.4, 
                'classProbThreshold': 0.4 }

    def detect_callback(self, info):
        self.detect_result.append(info)

    def image_detect(self, image, width=400, height=400, callback=None):
        if callback is None:
            callback = self.detect_callback
        
        self.detect = False
        def done_callback():
            self.detect = True
        if isinstance(image,str):
            self.js_init("""
                function handleResults(error, result) {
                    if(error){
                    console.error(error);
                    return;
                    }
                    console.log(result);
                    for (i=0;i<result.length;i++){
                        callback(result[i]);
                    }
                    done_callback();
                }
                var imageData = new Image(width, height)
                imageData.src = src;
                //console.log(imageData);
                element.predict_images = []
                element.predict_images.push(imageData);
                element.nn_info.network.detect(element.predict_images[0], handleResults);

            """, src=image, width=width, height=height,
                callback=callback, done_callback = done_callback)
            with ui_events() as poll:
                while self.detect is False:
                    poll(10)                # React to UI events (upto 10 at a time)
                    print('.', end='')
                    time.sleep(0.1)
            print('done')
        else:
            if isinstance(image,np.ndarray):
                if len(image.shape)==1:
                    if width*height!=image.shape[0]:
                        raise ValueError('image shape should be consistent with width and height')
                elif len(image.shape)==2:
                    raise ValueError("Please provide a rgba image pixel array")
                else:
                    if image.shape[2]!=4:
                        raise ValueError("Please provide a rgba image pixel array")
                    else:
                        image = image.flatten()
                image = image.tolist()
            self.js_init("""
                var canvas = document.createElement('canvas');
                canvas.width = width;
                canvas.height = height;
                var ctx = canvas.getContext('2d');
                var imgData=ctx.getImageData(0,0,width,height);
                imgData.data.set(d);
                function handleResults(error, result) {
                    if(error){
                    console.error(error);
                    return;
                    }
                    console.log(result);
                    for (i=0;i<result.length;i++){
                        callback(result[i]);
                    }
                    done_callback();
                }
                element.nn_info.network.detect(imgData, handleResults);
            """,d = image, width=width, height=height,
                callback=callback, done_callback = done_callback)
            with ui_events() as poll:
                while self.detect is False:
                    poll(10)                # React to UI events (upto 10 at a time)
                    print('.', end='')
                    time.sleep(0.1)
            print('done')

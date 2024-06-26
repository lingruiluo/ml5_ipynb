from . import ml5_nn
from . import utilis
import jp_proxy_widget
from IPython.display import display
from jupyter_ui_poll import ui_events
import numpy as np
import time


class imageClassifier(ml5_nn.neuralNetwork):

    def __init__(self, model=None, options=None, *pargs, **kwargs):
        super(imageClassifier,self).__init__(options=options,*pargs, **kwargs)
        self.data = []
        if options is None:
            options = self.default_options()
        if model is None:
            model = 'MobileNet'
        # self.model_path = model
        self.element.html("Loaded ml5.js")
        self.classify_callback_list = []
        self.classify_done = False
        self.model_load = False

        self.js_init("""
            element.nn_info = {};
            const image_model = ml5.imageClassifier(model, modelReady);
            element.nn_info.network = image_model;
            function modelReady() {
                console.log('Model Ready!');
                model_ready()
            }
            element.predict_images = [];
            let imageData;
        """,model = model,model_ready=self.model_ready)
        with ui_events() as poll:
            while self.model_load is False:
                poll(10)
                print('.', end='')
                time.sleep(0.1)
        print('Model is ready')
        time.sleep(0.05)
    
    def default_options(self):
        return {'version': 1,'alpha': 1.0,'topk': 3,}

    def model_ready(self):
        self.model_load = True

    def done_callback(self):
            self.classify_done = True
    
    def message(self,info):
            print(info)

    def classify_data(self, image, width=400, height=400, 
                    num_of_class = 3, callback=None):
        if callback is None:
            callback = self.classify_callback
        
        self.classify_done = False

        if isinstance(image,str):
            self.js_init("""
                function handleResults(error, result) {
                    if(error){
                    console.error(error);
                    return;
                    }
                    console.log(result);
                    callback(result);
                    done_callback();
                }
                imageData = new Image(width, height);
                imageData.src = src;
                message("image created");
                console.log(imageData);
                // element.predict_images = []
                // element.predict_images.push(imageData);
                setTimeout(function(){ 
                    element.nn_info.network.classify(imageData, num_of_class, handleResults);
                     }, 20);
                //element.nn_info.network.classify(imageData, num_of_class, handleResults);

            """, src=image, width=width, height=height,
                num_of_class = num_of_class,
                callback=callback, done_callback = self.done_callback,message=self.message)
            with ui_events() as poll:
                while self.classify_done is False:
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
                    callback(result);
                    done_callback();
                }
                element.nn_info.network.classify(imgData, num_of_class, handleResults);
            """,d = image, width=width, height=height,
                num_of_class = num_of_class,
                callback=callback, done_callback = self.done_callback)
            with ui_events() as poll:
                while self.classify_done is False:
                    poll(10)                # React to UI events (upto 10 at a time)
                    print('.', end='')
                    time.sleep(0.1)
            print('done')
    
    def keras_loadModel(self, path):
        self.model_load = False
        self.js_init("""
            const load_model = async () => {
                console.log("loading...");
                let my_model = await tf.loadLayersModel(path);
                element.nn_info.network = my_model;
            }
            load_model();
            console.log("loaded");
            model_ready();
        """,path=path, model_ready = self.model_ready)
        with ui_events() as poll:
            while self.model_load is False:
                poll(10)
                print('.', end='')
                time.sleep(0.1)
        print('Model is ready')

    
    def keras_predict(self, img, input_shape, callback=None):

        self.classify_done = False
        if callback is None:
            callback = self.classify_callback
        height, width, channel = input_shape
        if isinstance(img,str):
            self.js_init("""
                imageData = new Image(width, height);
                imageData.src = img_path;
                //imageData.crossOrigin = "anonymous";
                element.nn_info.image = imageData;

                                    let my_model = element.nn_info.network;
                    async function predict(imgElement, width, height, channel) {
                        console.log('Predicting...');
                        const startTime1 = performance.now();
                        let startTime2;

                        const logits = tf.tidy(() => {
                            // tf.browser.fromPixels() returns a Tensor from an image element.
                            const img_array = tf.browser.fromPixels(imgElement).toFloat();
                            console.log(imgElement);
                            console.log(img_array);
                            const normalized = img_array.div(255.0);
                            // Reshape to a single-element batch so we can pass it to predict.
                            const batched = normalized.reshape([1, width, height, channel]);

                            startTime2 = performance.now();
                            const prediction = my_model.predict(batched);
                            return prediction;
                        });
                        
                        async function getClasses(logits) {
                            const values = await logits.data();
                            console.log(values);
                            callback(values);
                            return values;
                        }

                        // Convert logits to probabilities and class names.
                        const classes = await getClasses(logits);
                        const totalTime1 = performance.now() - startTime1;
                        const totalTime2 = performance.now() - startTime2;
                        console.log(`Done in ${Math.floor(totalTime1)} ms ` +
                            `(not including preprocessing: ${Math.floor(totalTime2)} ms)`);
                        done_callback();
                    };

                    setTimeout(function(){ 
                        //predict(imageData, width, height, channel);
                        predict(element.nn_info.image, width, height, channel);
                    }, 50);
                    message("done");
            """, img_path = img, width = width, height = height, channel = channel,
             callback = callback, message = self.message, done_callback = self.done_callback)
        elif isinstance(img, np.ndarray):
            d = img.flatten().tolist()
            self.js_init("""
                    const arr = new Uint8ClampedArray(d.length);
                    // Iterate through every pixel
                    for (let i = 0; i < arr.length; i += 4) {
                    arr[i + 0] = d[i];    // R value
                    arr[i + 1] = d[i+1];  // G value
                    arr[i + 2] = d[i+2];    // B value
                    arr[i + 3] = d[i+3];  // A value
                    }
                    let imageData = new ImageData(arr, width);
                    console.log(imageData);
                    element.nn_info.image = imageData;

                    let my_model = element.nn_info.network;
                    async function predict(imgElement, width, height, channel) {
                        console.log('Predicting...');
                        const startTime1 = performance.now();
                        let startTime2;

                        const logits = tf.tidy(() => {
                            // tf.browser.fromPixels() returns a Tensor from an image element.
                            const img_array = tf.browser.fromPixels(imgElement).toFloat();
                            console.log(imgElement);
                            console.log(img_array);
                            const normalized = img_array.div(255.0);
                            // Reshape to a single-element batch so we can pass it to predict.
                            const batched = normalized.reshape([1, width, height, channel]);

                            startTime2 = performance.now();
                            const prediction = my_model.predict(batched);
                            return prediction;
                        });
                        
                        async function getClasses(logits) {
                            const values = await logits.data();
                            console.log(values);
                            callback(values);
                            return values;
                        }

                        // Convert logits to probabilities and class names.
                        const classes = await getClasses(logits);
                        const totalTime1 = performance.now() - startTime1;
                        const totalTime2 = performance.now() - startTime2;
                        console.log(`Done in ${Math.floor(totalTime1)} ms ` +
                            `(not including preprocessing: ${Math.floor(totalTime2)} ms)`);
                        done_callback();
                    };

                    setTimeout(function(){ 
                        //predict(imageData, width, height, channel);
                        predict(element.nn_info.image, width, height, channel);
                    }, 50);
                    message("done");
            """, d = d, width = width, height = height, channel = channel,
             callback = callback, message = self.message, done_callback = self.done_callback)
        
        # self.js_init("""
        #     let my_model = element.nn_info.network;
        #     async function predict(imgElement, width, height, channel) {
        #         console.log('Predicting...');
        #         const startTime1 = performance.now();
        #         let startTime2;

        #         const logits = tf.tidy(() => {
        #             // tf.browser.fromPixels() returns a Tensor from an image element.
        #             const img_array = tf.browser.fromPixels(imgElement).toFloat();
        #             console.log(imgElement);
        #             console.log(img_array);
        #             const normalized = img_array.div(255.0);
        #             // Reshape to a single-element batch so we can pass it to predict.
        #             const batched = normalized.reshape([1, width, height, channel]);

        #             startTime2 = performance.now();
        #             const prediction = my_model.predict(batched);
        #             return prediction;
        #         });
                
        #         async function getClasses(logits) {
        #             const values = await logits.data();
        #             console.log(values);
        #             callback(values);
        #             return values;
        #         }

        #         // Convert logits to probabilities and class names.
        #         const classes = await getClasses(logits);
        #         const totalTime1 = performance.now() - startTime1;
        #         const totalTime2 = performance.now() - startTime2;
        #         console.log(`Done in ${Math.floor(totalTime1)} ms ` +
        #             `(not including preprocessing: ${Math.floor(totalTime2)} ms)`);
        #         done_callback();
        #     };

        #     setTimeout(function(){ 
        #         //predict(imageData, width, height, channel);
        #         predict(element.nn_info.image, width, height, channel);
        #     }, 50);
        #     message("done");
        # """, width = width, height = height, channel = channel,
        #      callback = callback, message = self.message, done_callback = self.done_callback)
        with ui_events() as poll:
            while self.classify_done is False:
                poll(10)                # React to UI events (upto 10 at a time)
                print('.', end='')
                time.sleep(0.1)
        print('done')


class featureExtractor(ml5_nn.neuralNetwork):
    
    def __init__(self, task, model='MobileNet', options=None, 
                numLabels = None, learningRate = None, epochs = None, *pargs, **kwargs):
        super(featureExtractor,self).__init__(options=options,*pargs, **kwargs)
        self.element.html("Loaded ml5.js")
        self.classify_callback_list = []
        self.track = False
        self.model_load = False
        def model_ready():
            self.model_load = True
        self.options = options
        if options is None:
            self.options = self.default_options()
        if numLabels is not None:
            self.options['numLabels'] = numLabels
        if learningRate is not None:
            self.options['learningRate'] = learningRate
        if epochs is not None:
            self.options['epochs'] = epochs
        if task == 'classification':
            self.js_init("""
                element.nn_info = {};
                const fe = ml5.featureExtractor(model = model, callback = modelReady);
                const classifier = fe.classification(null, options);
                element.nn_info.network = classifier;
                function modelReady() {
                    console.log('Model Ready!');
                    model_ready()
                }
                let imageData;
            """,model = model,model_ready=model_ready, options = self.options)
        else:
            self.js_init("""
                element.nn_info = {};
                const fe = ml5.featureExtractor(model = model, callback = modelReady);
                const regressor = fe.regression();
                element.nn_info.network = regressor;
                function modelReady() {
                    console.log('Model Ready!');
                    model_ready()
                }
                let imageData;
            """,model = model,model_ready=model_ready)
        with ui_events() as poll:
            while self.model_load is False:
                poll(10)
                print('.', end='')
                time.sleep(0.1)
        print('Model is ready')
        time.sleep(0.05)
    
    def default_options(self):
        return {
            'version': 1,
            'alpha': 1.0,
            'topk': 3,
            'learningRate': 0.0001,
            'hiddenUnits': 100,
            'epochs': 20,
            'numLabels': 2,
            'batchSize': 0.4,
            }

    def done_callback(self):
        self.track = True

    def add_image(self, img, label, width=299, height=299):
        self.track = False

        if isinstance(img,str):
            self.js_init("""
                function image_added() {
                    console.log("added");
                    done_callback();
                }
                imageData = new Image(width, height);
                imageData.src = src;
                console.log(imageData);
                setTimeout(function(){ 
                    element.nn_info.network.addImage(imageData, label, image_added);
                     }, 30);
            """, src=img, 
                 width=width, height=height,
                 label = label,
                 done_callback=self.done_callback)
        else:
            if isinstance(img,np.ndarray):
                if len(img.shape)==1:
                    if width*height!=img.shape[0]:
                        raise ValueError('image shape should be consistent with width and height')
                elif len(img.shape)==2:
                    raise ValueError("Please provide a rgba image pixel array")
                else:
                    if img.shape[2]!=4:
                        raise ValueError("Please provide a rgba image pixel array")
                    else:
                        img = img.flatten()
                img = img.tolist()
            self.js_init("""
                var canvas = document.createElement('canvas');
                canvas.width = width;
                canvas.height = height;
                var ctx = canvas.getContext('2d');
                var imgData=ctx.getImageData(0,0,width,height);
                imgData.data.set(d);
                
                function image_added() {
                    console.log("added");
                    done_callback();
                }
                element.nn_info.network.addImage(imgData, label, image_added);
            """,d = img, width=width, height=height,
                label = label,
                done_callback = self.done_callback)
        with ui_events() as poll:
            while self.track is False:
                poll(10)                # React to UI events (upto 10 at a time)
                print('.', end='')
                time.sleep(0.1)
        print('done')
    
    def train(self):
        self.track = False
        def message(info):
            print(info)

        self.js_init("""
            if (element.nn_info.network.hasAnyTrainedClass){
                element.nn_info.network.train((lossValue) => {
                    console.log('Loss is', lossValue);
                });
            } else {
                message("No new data added");
            }
            done_callback();
        """, message=message,
             done_callback=self.done_callback)
        with ui_events() as poll:
            while self.track is False:
                poll(10)                # React to UI events (upto 10 at a time)
                print('.', end='')
                time.sleep(0.1)
        print('done')

    def classify(self, img, width=299, height=299,callback=None):
        
        if callback is None:
            callback = self.classify_callback
        def classify_callback(info):
            self.classify_callback_list.append(info)
        if isinstance(img,str):
            self.js_init("""
                new_image = new Image(width, height);
                new_image.src = src;
                setTimeout(function(){ 
                    element.nn_info.network.classify(new_image, (err, result) => {
                        callback(result); 
                        console.log(result); 
                        done_callback();
                    })}, 30);
            """, src=img, 
                 width=width, height=height, 
                 callback = callback,
                 done_callback=self.done_callback)
        else:
            if isinstance(img,np.ndarray):
                if len(img.shape)==1:
                    if width*height!=img.shape[0]:
                        raise ValueError('image shape should be consistent with width and height')
                elif len(img.shape)==2:
                    raise ValueError("Please provide a rgba image pixel array")
                else:
                    if img.shape[2]!=4:
                        raise ValueError("Please provide a rgba image pixel array")
                    else:
                        img = img.flatten()
                img = img.tolist()
            self.js_init("""
                var canvas = document.createElement('canvas');
                canvas.width = width;
                canvas.height = height;
                var ctx = canvas.getContext('2d');
                var imgData=ctx.getImageData(0,0,width,height);
                imgData.data.set(d);
                
                setTimeout(function(){ 
                    element.nn_info.network.classify(imgData, (err, result) => {
                        console.log(result);
                        callback(result); 
                        done_callback();
                    })}, 30);
            """,d = img, width=width, height=height,
                callback = callback,
                done_callback = self.done_callback)
        with ui_events() as poll:
            while self.track is False:
                poll(10)                # React to UI events (upto 10 at a time)
                print('.', end='')
                time.sleep(0.1)
        print('done')
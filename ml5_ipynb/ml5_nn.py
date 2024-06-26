"""
Framework for wrapping 
"""

import jp_proxy_widget
from IPython.display import display
from jupyter_ui_poll import ui_events
import time


def load_requirements(widget=None, silent=True, additional=()):
    """
    Load Javascript prerequisites into the notebook page context.
    """
    if widget is None:
        widget = jp_proxy_widget.JSProxyWidget()
        silent = False
    # Make sure jQuery and jQueryUI are loaded.
    widget.check_jquery()
    # load additional jQuery plugin code.
    ml5_js = ["https://unpkg.com/ml5@0.6.0/dist/ml5.js",
              "https://cdn.jsdelivr.net/npm/p5@1.1.9/lib/p5.js",
              "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.0.0/dist/tf.min.js"]
    widget.load_js_files(ml5_js)
    if not silent:
        widget.element.html("<div>Requirements for <b>chart_ipynb</b> have been loaded.</div>")
        display(widget)

class neuralNetwork(jp_proxy_widget.JSProxyWidget):
    

    def __init__(self, options=None, *pargs, **kwargs):
        super(neuralNetwork, self).__init__(*pargs, **kwargs)
        load_requirements(self)
        display(self.debugging_display())
        self.element.html("Loaded ml5.js")
        if options is None:
            options = self.default_options()
        self.options = options
        self.classify_callback_list = []
        self.predict_callback_list = []
        self.train_done = False
        self.classify_done = False
        self.predict_done = False
        self.model_created = False
        self.model_load = False

    def default_options(self):
        return {
            'inputs': [],
            'outputs': [],
            'dataUrl': None,
            'modelUrl': None,
            'task': None,
            'debug': False,
            'learningRate': 0.2,
            'hiddenUnits': 16,
        }

    def add_layer(self, layer):
        if 'layers' in self.options:
            self.options['layers'].append(layer)
        else:
            self.options['layers'] = [layer]

    def initialize_framework(self, options=None):
        if options is None:
            options = self.options
        def model_create():
            self.model_created = True
        self.js_init("""
            const nn = ml5.neuralNetwork(options);
            console.log("create network done!");
            element.nn_info = {
                network: nn };
            model_create();
        """, options = options, model_create=model_create)
        with ui_events() as poll:
            while self.model_created is False:
                poll(10)
                print('.', end='')
                time.sleep(0.1)
        print('Model is created')
        time.sleep(0.05)

    def add_data(self, inputs, outputs):
        self.js_init("""
            //debugger;
            element.nn_info.network.addData(inputs, outputs);

            //console.log(element.nn_info.network.data.data);
        """, inputs = inputs, outputs = outputs)

    
    def normalize_data(self):
        self.js_init("""
            element.nn_info.network.normalizeData();
            //console.log(element.nn_info.network.data.data);
        """)

    def train_data(self, trainingOptions=None):
        self.train_done = False
        def done_callback():
            self.train_done = True
        
        self.js_init("""
            function whileTraining(epoch, loss) {
                //console.log(epoch);
                console.log(loss);
                console.log(`epoch: ${epoch}, loss:${loss}`);
            }
            function doneTraining() {
                console.log('done!');
                done_callback();
            }
            //debugger;
            element.nn_info.network.train(trainingOptions,whileTraining, doneTraining);


        """,trainingOptions = trainingOptions, done_callback =done_callback)
        with ui_events() as poll:
            while self.train_done is False:
                poll(10)                # React to UI events (upto 10 at a time)
                print('.', end='')
                time.sleep(0.1)
        print('done')

    def done_callback(self):
        ##
        print("done!")

    def classify_data(self, input, callback=None):
        if callback is None:
            callback = self.classify_callback
        self.classify_done = False
        def done_callback():
            self.classify_done = True
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
            element.nn_info.network.classify(input, handleResults);

        """, input=input, callback=callback, done_callback = done_callback)
        with ui_events() as poll:
            while self.classify_done is False:
                poll(10)                # React to UI events (upto 10 at a time)
                print('.', end='')
                time.sleep(0.1)
        print('done')
    
    def classify_callback(self, info):
        self.classify_callback_list.append(info)


    def predict_data(self, input, callback=None):
        if callback is None:
            callback = self.predict_callback
        self.predict_done = True
        def done_callback():
            self.predict_done = True
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
            element.nn_info.network.predict(input, handleResults);

        """, input=input, callback=callback, done_callback=done_callback)
        with ui_events() as poll:
            while self.predict_done is False:
                poll(10)                # React to UI events (upto 10 at a time)
                print('.', end='')
                time.sleep(0.1)
        print('done')

    def predict_callback(self, info):
        self.predict_callback_list.append(info)


    def save(self, output_name='model'):
        self.js_init("""
            function model_saved(){
                console.log("model saved");
            }
            element.nn_info.network.save(output_name, model_saved);
        """, output_name=output_name)
    
    def load(self, model, metadata=None, weights=None, modelLoaded=None):
        self.model_load = False
        def default_modelLoaded():
            print("model loaded")
        
        def check_load():
            self.model_load = True
        if modelLoaded is None:
            modelLoaded = default_modelLoaded
        modelDetails = model
        if metadata is not None and weights is not None:
            modelDetails = {
                'model': model,
                'metadata': metadata,
                'weights': weights
            }


        self.js_init("""
            function modelLoaded_callback(){
                modelLoaded();
                check_load();
                console.log("model loaded");
            }
            element.nn_info.network.load(modelDetails, modelLoaded_callback);
        """, modelDetails = modelDetails, 
             modelLoaded=modelLoaded,
             check_load = check_load)
        with ui_events() as poll:
            while self.model_load is False:
                poll(10)
                print('.', end='')
                time.sleep(0.1)
        print('done')
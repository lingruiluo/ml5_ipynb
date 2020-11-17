import jp_proxy_widget
from . import ml5_nn
from IPython.display import display
from jupyter_ui_poll import ui_events
import time


class word2Vec(ml5_nn.neuralNetwork):
    
    def __init__(self, model, options=None, *pargs, **kwargs):
        super(word2Vec,self).__init__(options=options,*pargs, **kwargs)
        if options is None:
            options = self.default_options()
        self.element.html("Loaded ml5.js")
        # if model is None:
        #     model = 'data/wordvecs10000.json'
        
        self.add_results = []
        self.subtract_results = []
        self.average_results = []
        self.nearest_results = []
        self.nearestSet_results = []
        self.model_load = False
        self.track = False
        def model_ready():
            self.model_load = True

        self.js_init("""
            element.nn_info = {};
            const w2v = ml5.word2vec(model, modelReady);
            element.nn_info.network = w2v;
            function modelReady() {
                console.log('Model Ready!');
                model_ready()
            }
        """,model = model, model_ready=model_ready)
        with ui_events() as poll:
            while self.model_load is False:
                poll(10)
                print('.', end='')
                time.sleep(0.1)
        print('Model is ready')
        time.sleep(0.05)

    def add(self, words, callback=None):
        self.track = False
        def add_callback(results):
            self.add_results.append(results)
            self.track = True
        if callback is None:
            callback = add_callback
        self.js_init("""
            element.nn_info.network.add(words, (err, results)=>{
                if(err){
                    console.error(err);
                    return;
                }
                console.log(results);
                callback(results);
            });
            
        """, words = words, callback = callback)
        with ui_events() as poll:
            while self.track is False:
                poll(10)                # React to UI events (upto 10 at a time)
                print('.', end='')
                time.sleep(0.1)
        print('done')

    def subtract(self, words, callback=None):
        self.track = False
        def subtract_callback(results):
            self.subtract_results.append(results)
            self.track = True
        
        if callback is None:
            callback = subtract_callback
        self.js_init("""
            element.nn_info.network.subtract(words, (err, results)=>{
                if(err){
                    console.error(err);
                    return;
                }
                console.log(results);
                callback(results);
            });
        """, words = words, callback = callback)
        with ui_events() as poll:
            while self.track is False:
                poll(10)                # React to UI events (upto 10 at a time)
                print('.', end='')
                time.sleep(0.1)
        print('done')
    
    def average(self, words, callback = None):
        self.track = False
        def average_callback(results):
            self.average_results.append(results)
            self.track = True
        if callback is None:
            callback = average_callback

        self.js_init("""
            element.nn_info.network.average(words, (err, results)=>{
                if(err){
                    console.error(err);
                    return;
                }
                console.log(results);
                callback(results);
            });
        """, words = words, callback = callback)
        with ui_events() as poll:
            while self.track is False:
                poll(10)                # React to UI events (upto 10 at a time)
                print('.', end='')
                time.sleep(0.1)
        print('done')

    def nearest(self, word, callback = None):
        self.track = False
        def nearest_callback(results):
            self.nearest_results.append(results)
            self.track = True
        if callback is None:
            callback = nearest_callback

        self.js_init("""
            element.nn_info.network.nearest(word, (err, results)=>{
                if(err){
                    console.error(err);
                    return;
                }
                console.log(results);
                callback(results);
            });
        """, word = word, callback = callback)
        with ui_events() as poll:
            while self.track is False:
                poll(10)                # React to UI events (upto 10 at a time)
                print('.', end='')
                time.sleep(0.1)
        print('done')

    def nearest(self, word, callback = None):
        self.track = False
        def nearest_callback(results):
            self.nearest_results.append(results)
            self.track = True
        if callback is None:
            callback = nearest_callback

        self.js_init("""
            element.nn_info.network.nearest(word, (err, results)=>{
                if(err){
                    console.error(err);
                    return;
                }
                console.log(results);
                callback(results);
            });
        """, word = word, callback = callback)
        with ui_events() as poll:
            while self.track is False:
                poll(10)                # React to UI events (upto 10 at a time)
                print('.', end='')
                time.sleep(0.1)
        print('done')

    def nearestFromSet(self, word, word_set, callback = None):
        self.track = False
        def nearestSet_callback(results):
            self.nearestSet_results.append(results)
            self.track = True
        if callback is None:
            callback = nearestSet_callback

        self.js_init("""
            element.nn_info.network.nearestFromSet(word, word_set, (err, results)=>{
                if(err){
                    console.error(err);
                    return;
                }
                console.log(results);
                callback(results);
            });
        """, word = word, word_set = word_set, callback = callback)
        with ui_events() as poll:
            while self.track is False:
                poll(10)               
                print('.', end='')
                time.sleep(0.1)
        print('done')
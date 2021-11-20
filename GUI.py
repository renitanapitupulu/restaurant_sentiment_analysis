from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.config import Config
from data.engine import predict_review

WIDTH = '750'
HEIGHT = '600'

Config.set('graphics', 'width', WIDTH)
Config.set('graphics', 'height', HEIGHT)

Builder.load_string("""
<MainScreen>:
    id: main
    FloatLayout:
        canvas:
            Color:
                rgba: 1, 1, 1, 1
                
            Rectangle:
                source: 'background.jpg'
                size: self.size
                pos: self.pos

            RoundedRectangle:
                size: 700, 290
                pos: 24, 290
                radius: [30, 30, 30, 30]

            RoundedRectangle:
                size: 224, 260
                pos: 24, 20
                radius: [20, 20, 20, 20]
            
            RoundedRectangle:
                size: 224, 260
                pos: 264, 20
                radius: [20, 20, 20, 20]
            
            RoundedRectangle:
                size: 224, 260
                pos: 504, 20
                radius: [20, 20, 20, 20]

        Label:
            text: 'Restaurant review:'
            font_size: 20
            pos: -240, 250
            bold: True
            color: 0, 0, 0, 1
        
        TextInput:
            id: review
            hint_text: 'Enter restaurant review'
            pos: 50, 318
            size_hint: 0.6, 0.35

        # Button:
        #     text: 'Extract Aspect'
        #     size_hint: 0.2, 0.1
        #     pos: 540, 465
        #     background_normal: ''
        #     background_color: 0.98, 0.733, 0.188, 1
        #     on_press: main.extractAspect()

        Button:
            text: 'Analyze'
            size_hint: 0.2, 0.1
            pos: 540, 395
            background_normal: ''
            background_color: 0.98, 0.733, 0.188, 1
            on_press: main.analyze()

        # Button:
        #     text: 'Classify Sentiment'
        #     size_hint: 0.2, 0.1
        #     pos: 540, 325
        #     background_normal: ''
        #     background_color: 0.98, 0.733, 0.188, 1
        #     on_press: main.classifySentiment()

        Label:
            text: 'Aspect'
            font_size: 20
            pos: -242, -40
            bold: True
            color: 0, 0, 0, 1

        Label:
            id: aspect
            size: self.texture_size
            font_size : 18
            pos: -242, -150
            color: 0, 0, 0, 1

        Label:
            text: 'Category'
            font_size: 20
            pos: -2, -40
            bold: True
            color: 0, 0, 0, 1
        
        Label:
            id: category
            size: self.texture_size
            font_size : 18
            pos: -2, -150
            color: 0, 0, 0, 1

        Label:
            text: 'Polarity'
            font_size: 20
            pos: 238, -40
            bold: True
            color: 0, 0, 0, 1
        
        Label:
            id: polarity
            size: self.texture_size
            font_size : 18
            pos: 238, -150
            color: 0, 0, 0, 1

        Image:
            id: icon
            pos: 238, -160
            size: 100, 100
""")


class MainScreen(Screen):
    def extractAspect(self):
        review = self.ids['review'].text
        self.ids['aspect'].text = 'Test aspect'

    def analyze(self):
        review = self.ids['review'].text
        aspect, category, sentiment = predict_review(review)
        aspectText = ''
        for a in aspect:
            aspectText += a + '\n'
        self.ids['aspect'].text = aspectText
        categoryText = ''
        for c in category:
            categoryText += c + '\n'
        self.ids['category'].text = categoryText
        sentimentText = ''
        for pair in sentiment.items():
            ctgr = pair[0]
            if (pair[1] == -1):
                polarity = 'Negative'
            elif (pair[1] == 1):
                polarity = 'Positive'
            else:
                polarity = 'Neutral'
            sentimentText += pair[0] + ': ' + polarity + '\n'
        self.ids['polarity'].text = sentimentText

    def classifySentiment(self):
        review = self.ids['review'].text
        self.ids['polarity'].text = 'Test polarity'
        self.ids['icon'].source = 'positive.png'


sm = ScreenManager()
sm.add_widget(MainScreen(name='main'))


class MyApp(App):
    def build(self):
        self.title = 'Restaurant Analysis Sentiment'
        return sm


MyApp().run()
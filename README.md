# Guide to start running this locally

Download all the required packages:

``` 
pip install -r requirements.txt
```

Run the server:
```
python server.py
```

Go to URL "http://0.0.0.0:8000", upload your image and get the prediction.

---
To train this model locally on your machine, you could execute following command:
```
python model.py
```
---

I have only used 1000 images from each class (FAKE and REAL) from the original dataset to train this model. You could also try training it on more images. To download the original dataset click [here](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images).
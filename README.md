# tianchi_game

* images:0.07: attack resnet ,no rorate, eps=16/255,steps=20,le=2e-1 失败

* images:0.10: attack resnet ,no rorate, eps=16/255,steps=20,le=2e-1 并将tensorflow升级到1.4.0版本，score:111.0310
* images:0.12: attack resnet and vgg ,no rotate ,eps=16/255,steps=20,le=2e-1,score：108.4180
* images:0.13: attack resnet and vgg ,no rotate ,eps=16/255,steps=40,le=2e-1,score：107.5490
* images:0.14: attack resnet and vgg ,no rotate ,eps=32/255,steps=40,le=2e-1,score：107.4120
* images:0.15: attack reset,vgg and inception,eps=16/255,steps=50,lr=3e-1 ,logits average ,score：105.0430
* images:0.16: attack reset,vgg and inception,eps=16/255,steps=50,lr=5e-1 ,logits average ,score：103.8740
* images:0.17: attack reset,vgg and inception,eps=32/255,steps=80,lr=6e-1 ,logits average ,score：

# tianchi_game

* images:0.07: attack resnet ,no rorate, eps=16/255,steps=20,le=2e-1 失败

* images:0.10: attack resnet ,no rorate, eps=16/255,steps=20,le=2e-1 并将tensorflow升级到1.4.0版本，score:111.0310
* images:0.12: attack resnet and vgg ,no rotate ,eps=16/255,steps=20,le=2e-1,score：108.4180
* images:0.13: attack resnet and vgg ,no rotate ,eps=16/255,steps=40,le=2e-1,score：107.5490
* images:0.14: attack resnet and vgg ,no rotate ,eps=32/255,steps=40,le=2e-1,score：107.4120
* images:0.15: attack resnet,vgg and inception,eps=16/255,steps=50,lr=3e-1 ,logits average ,score：105.0430
* images:0.16: attack resnet,vgg and inception,eps=16/255,steps=50,lr=5e-1 ,logits average ,score：103.8740
* images:0.17: attack resnet,vgg and inception,eps=32/255,steps=80,lr=6e-1 ,logits average ,score：100.0830
* images:0.18: attack resnet,vgg and inception,use cleverhans MomentumIterativeMethod 
attack_params = {"eps": 32.0 / 255.0, "eps_iter": 0.01, "clip_min": -1.0, "clip_max": 1.0, \
                             "nb_iter": 20, "decay_factor": 1.0, "y_target": one_hot_target_class}
                             score: 92.0006
* images:0.19: attack resnet,vgg and inception,use cleverhans MomentumIterativeMethod 
attack_params = {"eps": 32.0 / 255.0, "eps_iter": 0.01, "clip_min": -1.0, "clip_max": 1.0, \
                             "nb_iter": 15, "decay_factor": 1.0, "y_target": one_hot_target_class}
                             score:88.4346
* images:0.20: attack resnet,vgg and inception,use cleverhans MomentumIterativeMethod 
attack_params = {"eps": 40.0 / 255.0, "eps_iter": 0.01, "clip_min": -1.0, "clip_max": 1.0, \
                             "nb_iter": 15, "decay_factor": 1.0, "y_target": one_hot_target_class}
                             score:87.9618
* images:0.21: attack resnet,vgg and inception,use cleverhans MomentumIterativeMethod 
attack_params = {"eps": 0.2, "eps_iter": 0.01, "clip_min": -1.0, "clip_max": 1.0, \
                             "nb_iter": 15, "decay_factor": 1.0, "y_target": one_hot_target_class}
                             score:87.1428

* 以下升级tensorflow到1.12.0，python为3.5，cleverhas为最新版。
* images:0.22: attack resnet,vgg and inception,use cleverhans MomentumIterativeMethod + ProjectedGradientDescent将对抗样本相加求均值,
attack_params = {"eps": 0.2, "eps_iter": 0.01, "clip_min": -1.0, "clip_max": 1.0, \
                             "nb_iter": 15, "decay_factor": 1.0, "y_target": one_hot_target_class}
attack_params2={"eps":0.2,"y_target":one_hot_target_class,"nb_iter":20,"clip_min":-1.0,"clip_max":1.0}
                            score:

* images:0.23: attack resnet,vgg and inception,use cleverhans MomentumIterativeMethod ，其输入图为0.7*原图+0.3*目标类的图
attack_params = {"eps": 0.2, "eps_iter": 0.01, "clip_min": -1.0, "clip_max": 1.0, \
                             "nb_iter": 15, "decay_factor": 1.0, "y_target": one_hot_target_class}
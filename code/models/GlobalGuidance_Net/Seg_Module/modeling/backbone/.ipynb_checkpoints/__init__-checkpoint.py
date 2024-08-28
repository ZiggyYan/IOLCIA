# from modeling.backbone import resnet, xception, drn, mobilenet,daf,daf_ds,dense,xception_glb,daf_ds_wrte
# from Seg_Module.modeling.backbone import xception
# from Seg_Module.modeling.backbone import daf_ds, drn, mobilenet, resnet, dense

from models.GlobalGuidance_Net.Seg_Module.modeling.backbone import xception
from models.GlobalGuidance_Net.Seg_Module.modeling.backbone import daf_ds, drn, mobilenet, resnet, dense

# import daf_ds, drn, mobilenet, resnet, dense
# import xception

def build_backbone(backbone, output_stride, BatchNorm):
    if backbone == 'resnet':
        return resnet.ResNet101(output_stride, BatchNorm)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm)
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm)
    elif backbone=='daf_ds':
        return daf_ds.daf_ds()
    elif backbone=='dense':
        return dense.Dense()
    # elif backbone == 'xception_glb':
    #     return xception_glb.AlignedXception(output_stride, BatchNorm)
    else:
        raise NotImplementedError

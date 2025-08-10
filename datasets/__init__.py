# from .blender import BlenderDataset
# from .satellite import SatelliteDataset
# from .satellite_depth import SatelliteDataset_depth
#
#
# def load_dataset(args, split):
#     outputs = []
#     if args.data == 'sat':
#         d1 = SatelliteDataset(root_dir=args.root_dir,
#                               img_dir=args.img_dir if args.img_dir is not None else args.root_dir,
#                               split=split,
#                               cache_dir=args.cache_dir,
#                               img_downscale=args.img_downscale)
#         outputs.append(d1)
#
#         if args.ds_lambda > 0 and split == 'train':
#             d2 = SatelliteDataset_depth(root_dir=args.root_dir,
#                                         img_dir=args.img_dir if args.img_dir is not None else args.root_dir,
#                                         split=split,
#                                         cache_dir=args.cache_dir,
#                                         img_downscale=args.img_downscale)
#             outputs.append(d2)
#     else:
#         outputs.append(BlenderDataset(root_dir=args.root_dir, split=split))
#
#     return outputs


from .satellite_dataset import TrainSatelliteDataset, ValidationSatelliteDataset


def load_train_dataset(args):
    outputs = []
    d1 = TrainSatelliteDataset(root_dir=args.root_dir,
                               img_dir=args.img_dir,
                               cache_dir=args.cache_dir,
                               img_downscale=args.img_downscale)
    outputs.append(d1)
    return outputs


def load_val_dataset(args):
    outputs = []
    d1 = ValidationSatelliteDataset(root_dir=args.root_dir,
                                    img_dir=args.img_dir,
                                    cache_dir=args.cache_dir,
                                    img_downscale=args.img_downscale)
    outputs.append(d1)
    return outputs

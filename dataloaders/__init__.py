# from dataloaders.datasets import cityscapes, coco, combine_dbs, pascal, sbd
# from torch.utils.data import DataLoader

# def make_data_loader(args, **kwargs):

#     if args.dataset == 'pascal':
#         train_set = pascal.VOCSegmentation(args, split='train')
#         val_set = pascal.VOCSegmentation(args, split='val')
#         if args.use_sbd:
#             sbd_train = sbd.SBDSegmentation(args, split=['train', 'val'])
#             train_set = combine_dbs.CombineDBs([train_set, sbd_train], excluded=[val_set])

#         num_class = train_set.NUM_CLASSES
#         train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
#         val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
#         test_loader = None

#         return train_loader, val_loader, test_loader, num_class
    
#     elif args.dataset == 'pascal_all':
#         trainval_set = pascal.VOCSegmentation(args, split='trainval')
#         if args.use_sbd:
#             sbd_all = sbd.SBDSegmentation(args, split=['train', 'val'])
#             trainval_set = combine_dbs.CombineDBs([trainval_set, sbd_all], excluded=None)

#         num_class = trainval_set.NUM_CLASSES
#         trainval_loader = DataLoader(trainval_set, batch_size=args.batch_size, shuffle=False, **kwargs)

#         return trainval_loader, num_class

#     elif args.dataset == 'cityscapes':
#         train_set = cityscapes.CityscapesSegmentation(args, split='train')
#         val_set = cityscapes.CityscapesSegmentation(args, split='val')
#         test_set = cityscapes.CityscapesSegmentation(args, split='test')
#         num_class = train_set.NUM_CLASSES
#         train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
#         val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
#         test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

#         return train_loader, val_loader, test_loader, num_class

#     elif args.dataset == 'coco':
#         train_set = coco.COCOSegmentation(args, split='train')
#         val_set = coco.COCOSegmentation(args, split='val')
#         num_class = train_set.NUM_CLASSES
#         train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
#         val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
#         test_loader = None
#         return train_loader, val_loader, test_loader, num_class

#     else:
#         raise NotImplementedError

# def make_data_loader_masked(args, **kwargs):

#     if args.dataset == 'pascal':
#         train_set = pascal.VOCSegmentationMasked(args, split='train')
#         val_set = pascal.VOCSegmentationMasked(args, split='val')
#         if args.use_sbd:
#             sbd_train = sbd.SBDSegmentationMasked(args, split=['train', 'val'])
#             train_set = combine_dbs.CombineDBs([train_set, sbd_train], excluded=[val_set])

#         num_class = train_set.NUM_CLASSES
#         train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
#         val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
#         test_loader = None

#         return train_loader, val_loader, test_loader, num_class

#     else:
#         raise NotImplementedError

# def make_data_loader_single(args, **kwargs):

#     if args.dataset == 'pascal':
#         train_set = pascal.VOCSegmentationMasked(args, split='train', not_masked=True)
#         val_set = pascal.VOCSegmentationMasked(args, split='val', not_masked=True)
#         if args.use_sbd:
#             sbd_train = sbd.SBDSegmentationMasked(args, split=['train', 'val'], not_masked=True)
#             train_set = combine_dbs.CombineDBs([train_set, sbd_train], excluded=[val_set])

#         num_class = train_set.NUM_CLASSES
#         train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
#         val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
#         test_loader = None

#         return train_loader, val_loader, test_loader, num_class

#     else:
#         raise NotImplementedError


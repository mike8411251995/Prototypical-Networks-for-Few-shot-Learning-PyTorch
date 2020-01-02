class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal' or dataset == 'pascal_all':
            return './datasets/VOC/VOCdevkit/VOC2012/'  # folder that contains VOCdevkit/.
        elif dataset == 'sbd':
            return './datasets/SBD/benchmark_RELEASE/'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return './datasets/cityscapes/'     # folder that contains leftImg8bit/
        elif dataset == 'coco':
            return './datasets/coco/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError

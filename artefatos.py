def confusão_direto(Y_pred, Y_val):

    FP = len(np.where(Y_pred - Y_val  == 255)[0])
    FN = len(np.where(Y_pred - Y_val  == -255)[0])
    TP = len(np.where(Y_pred + Y_val == 510)[0])
    TN = len(np.where(Y_pred + Y_val == 0)[0])

    cmat = [[TP, FN], [FP, TN]]
    
    
    return cmat


def metric_randANDvarInformation(Y_pred, Y_outro):
    method_names = ['Proposed',"Canny"]

    precision_list = []
    recall_list = []
    split_list = []
    merge_list = []

    for name, im_test in zip(method_names, [Y_pred, Y_outro]):
        error, precision, recall = adapted_rand_error(Y_val, im_test)
        splits, merges = variation_of_information(Y_val, im_test)
        split_list.append(splits)
        merge_list.append(merges)
        precision_list.append(precision)
        recall_list.append(recall)
        print(f'\n## Method: {name}')
        print(f'Adapted Rand error: {error}')
        print(f'Adapted Rand precision: {precision}')
        print(f'Adapted Rand recall: {recall}') 
        print(f'False Splits: {splits}')
        print(f'False Merges: {merges}')


####################

#segmap = file['groundTruth'][0][0][0][0][0]



#segmap_uint8 = segmap.astype(np.uint8) # convert to uint8 to prevent lossy conversion when saving image
#imageio.imwrite(os.path.join('_seg.jpg'), segmap_uint8)
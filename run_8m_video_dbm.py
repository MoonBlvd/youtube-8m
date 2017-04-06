import numpy as np
import tensorflow as tf
from tensorflow import gfile
from yadlt.models.boltzmann import dbn
from yadlt.utils import datasets, utilities

# #################### #
#   Flags definition   #
# #################### #
flags = tf.app.flags
FLAGS = flags.FLAGS

# Global configuration
flags.DEFINE_string('train_data_pattern', '', 'Pattern of train file.')
flags.DEFINE_string('eval_data_pattern', '', 'Pattern of validation file.')
flags.DEFINE_string('feature_name', '', 'feature extraction name')
flags.DEFINE_string('name', 'dbn', 'Name of the model.')
flags.DEFINE_string('save_predictions', '', 'Path to a .npy file to save predictions of the model.')
flags.DEFINE_string('save_layers_output_test', '', 'Path to a .npy file to save test set output from all the layers of the model.')
flags.DEFINE_string('save_layers_output_train', '', 'Path to a .npy file to save train set output from all the layers of the model.')
flags.DEFINE_boolean('do_pretrain', True, 'Whether or not pretrain the network.')
flags.DEFINE_integer('seed', -1, 'Seed for the random generators (>= 0). Useful for testing hyperparameters.')
flags.DEFINE_float('momentum', 0.5, 'Momentum parameter.')

# RBMs layers specific parameters
flags.DEFINE_string('rbm_layers', '256,', 'Comma-separated values for the layers in the sdae.')
flags.DEFINE_boolean('rbm_gauss_visible', False, 'Whether to use Gaussian units for the visible layer.')
flags.DEFINE_float('rbm_stddev', 0.1, 'Standard deviation for Gaussian visible units.')
flags.DEFINE_string('rbm_learning_rate', '0.001,', 'Initial learning rate.')
flags.DEFINE_string('rbm_num_epochs', '10,', 'Number of epochs.')
flags.DEFINE_string('rbm_batch_size', '32,', 'Size of each mini-batch.')
flags.DEFINE_string('rbm_gibbs_k', '1,', 'Gibbs sampling steps.')

# Supervised fine tuning parameters
flags.DEFINE_string('finetune_act_func', 'relu', 'Activation function.')
flags.DEFINE_float('finetune_learning_rate', 0.01, 'Learning rate.')
flags.DEFINE_float('finetune_momentum', 0.9, 'Momentum parameter.')
flags.DEFINE_integer('finetune_num_epochs', 10, 'Number of epochs.')
flags.DEFINE_integer('finetune_batch_size', 32, 'Size of each mini-batch.')
flags.DEFINE_string('finetune_opt', 'momentum', '["sgd", "ada_grad", "momentum", "adam"]')
flags.DEFINE_string('finetune_loss_func', 'softmax_cross_entropy', 'Loss function. ["mse", "softmax_cross_entropy"]')
flags.DEFINE_float('finetune_dropout', 1, 'Dropout parameter.')

# Conversion of Autoencoder layers parameters from string to their specific type
rbm_layers = utilities.flag_to_list(FLAGS.rbm_layers, 'int')
rbm_learning_rate = utilities.flag_to_list(FLAGS.rbm_learning_rate, 'float')
rbm_num_epochs = utilities.flag_to_list(FLAGS.rbm_num_epochs, 'int')
rbm_batch_size = utilities.flag_to_list(FLAGS.rbm_batch_size, 'int')
rbm_gibbs_k = utilities.flag_to_list(FLAGS.rbm_gibbs_k, 'int')

# Parameters validation
assert FLAGS.feature_name in ['mean_rgb', 'mean_audio']
assert FLAGS.finetune_act_func in ['sigmoid', 'tanh', 'relu']
assert len(rbm_layers) > 0

def read_video_level_data(feature_name, file_name_queue):
    features = []
    labels = []
    for video_level_record in file_name_queue:
        for example in tf.python_io.tf_record_iterator(video_level_record):
            tf_example = tf.train.Example.FromString(example)
            labels.append(tf_example.features.feature['labels'].int64_list.value)
            features.append(tf_example.features.feature[feature_name].float_list.value)
    return np.array(labels), np.array(feature_name)

if __name__ == '__main__':

    utilities.random_seed_np_tf(FLAGS.seed)
    tr_file_queue = gfile.Glob(FLAGS.train_data_pattern)
    vl_file_queue = gfile.Glob(FLAGS.eval_data_pattern)

    trY, trX = read_video_level_data(FLAGS.feature_name, tr_file_queue)
    vlY, vlX = read_video_level_data(FLAGS.feature_name, vl_file_queue)

    # Create the object
    finetune_act_func = utilities.str2actfunc(FLAGS.finetune_act_func)

    srbm = dbn.DeepBeliefNetwork(
        name=FLAGS.name, do_pretrain=FLAGS.do_pretrain,
        rbm_layers=rbm_layers,
        finetune_act_func=finetune_act_func, rbm_learning_rate=rbm_learning_rate,
        rbm_num_epochs=rbm_num_epochs, rbm_gibbs_k = rbm_gibbs_k,
        rbm_gauss_visible=FLAGS.rbm_gauss_visible, rbm_stddev=FLAGS.rbm_stddev,
        momentum=FLAGS.momentum, rbm_batch_size=rbm_batch_size, finetune_learning_rate=FLAGS.finetune_learning_rate,
        finetune_num_epochs=FLAGS.finetune_num_epochs, finetune_batch_size=FLAGS.finetune_batch_size,
        finetune_opt=FLAGS.finetune_opt, finetune_loss_func=FLAGS.finetune_loss_func,
        finetune_dropout=FLAGS.finetune_dropout)

    # Fit the model (unsupervised pretraining)
    if FLAGS.do_pretrain:
        srbm.pretrain(trX, vlX)

    # finetuning
    print('Start deep belief net finetuning...')
    srbm.fit(trX, trY, vlX, vlY)

    # Test the model
    #print('Test set accuracy: {}'.format(srbm.score(teX, teY)))

    # Save the predictions of the model
    if FLAGS.save_predictions:
        print('Saving the predictions for the test set...')
        np.save(FLAGS.save_predictions, srbm.predict(teX))


    def save_layers_output(which_set):
        if which_set == 'train':
            trout = srbm.get_layers_output(trX)
            for i, o in enumerate(trout):
                np.save(FLAGS.save_layers_output_train + '-layer-' + str(i + 1) + '-train', o)

        elif which_set == 'test':
            teout = srbm.get_layers_output(teX)
            for i, o in enumerate(teout):
                np.save(FLAGS.save_layers_output_test + '-layer-' + str(i + 1) + '-test', o)


    # Save output from each layer of the model
    if FLAGS.save_layers_output_test:
        print('Saving the output of each layer for the test set')
        save_layers_output('test')

    # Save output from each layer of the model
    if FLAGS.save_layers_output_train:
        print('Saving the output of each layer for the train set')
        save_layers_output('train')
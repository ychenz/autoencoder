# Created by Yuchen on 4/23/17.
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from SSAE import SSAE
import time


def main():
    mnist = input_data.read_data_sets("MINIST_data/ ", one_hot=True)
    with tf.Graph().as_default():
        with tf.name_scope("Train"):
            ssae_train = SSAE([784, 342], 128, lr=0.001, is_training=True, beta=0)

        with tf.name_scope("Encode"):
            ssae_encode = SSAE([784, 342], 1, is_training=False)

        save_path = '/home/tina/Scripts/python/Autoencoder/checkpoints/model.ckpt'
        sv = tf.train.Supervisor(logdir=save_path, summary_op=None)
        with sv.managed_session() as session:
            batch_cnt = 0
            start_time = time.time()
            ttl_loss = 0
            for step in range(100000):
                batch_xs, batch_ys = mnist.train.next_batch(128)
                feed_dict = {ssae_train.inputs: batch_xs}
                _, loss = session.run([ssae_train.train_op, ssae_train.loss], feed_dict=feed_dict)
                ttl_loss += loss
                batch_cnt += 1

                if step % 1000 == 0:
                    print('training cost at step %d: %.2f speed: %.0f bpm' %
                          (step, ttl_loss / batch_cnt, batch_cnt / float((time.time() - start_time) / 60.0)))
                    summaries = session.run(ssae_train.summary_op, feed_dict=feed_dict)
                    sv.saver.save(session, save_path)
                    sv.summary_computed(session, summaries)

            sv.saver.save(session, save_path)
            sv.summary_computed(session, summaries)

if __name__ == '__main__':
    main()

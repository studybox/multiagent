import numpy as np
import tensorflow as tf

import rl.common.tf_util as U
from rl.acktr.utils import conv, fc, dense, conv_to_fc, sample, kl_div, graphlayer, graphblock

# Graph network
class CategoricalGraphPolicy(object):
    def __init__(self, sess, obe_space, obn_space, ac_space,
                 nenv, nsteps, nstack, nedges, nnodes, reuse=False, name='model'):
        nbatch = nenv * nsteps
        obns_shape = (nbatch*nedges, obn_space * nstack)
        all_obns_shape = obns_shape
        obnr_shape = (nbatch*nedges, obn_space * nstack)
        all_obnr_shape = obnr_shape
        obe_shape = (nbatch*nedges, obe_space * nstack)
        all_obe_shape = obe_shape
        obn_shape = (nbatch*nnodes, obn_space * nstack)
        all_obn_shape = obn_shape
        nbnn = nbatch*nnodes if nnodes else None
        nbne = nbatch*nedges if nedges else None

        nact = ac_space.n
        all_ac_shape = (nbatch, nact*nstack)
        #all_ac_shape = (nbatch, (sum([ac.n for ac in ac_spaces]) - nact) * nstack)
        nfs = efs = 12
        X_ns = tf.placeholder(tf.float32, obns_shape)
        X_nr = tf.placeholder(tf.float32, obnr_shape)
        X_e = tf.placeholder(tf.float32, obe_shape)
        X_n = tf.placeholder(tf.float32, obn_shape)
        e2ns = tf.placeholder(tf.float32, (nbne, nbnn))
        e2nr = tf.placeholder(tf.float32, (nbne, nbnn))
        ns2e = tf.placeholder(tf.float32, (nbnn, nbne))
        b2e = tf.placeholder(tf.float32, (nbatch, nbne))
        b2n = tf.placeholder(tf.float32, (nbatch, nbnn))

        X_ns_v = tf.placeholder(tf.float32, all_obns_shape)
        X_nr_v = tf.placeholder(tf.float32, all_obnr_shape)
        X_e_v = tf.placeholder(tf.float32, all_obe_shape)
        X_n_v = tf.placeholder(tf.float32, all_obn_shape)
        e2ns_v = tf.placeholder(tf.float32, (nbne, nbnn))
        e2nr_v = tf.placeholder(tf.float32, (nbne, nbnn))
        ns2e_v = tf.placeholder(tf.float32, (nbnn, nbne))
        b2e_v = tf.placeholder(tf.float32, (nbatch, nbne))
        b2n_v = tf.placeholder(tf.float32, (nbatch, nbnn))

        A_v = tf.placeholder(tf.float32, all_ac_shape)

        with tf.variable_scope('policy_{}'.format(name), reuse=reuse):
            x = [X_ns, X_nr, X_e, X_n, e2ns, e2nr, ns2e]
            g1 = graphlayer(x, 'gl1', 128, nfs, efs, init_scale=np.sqrt(2))
            g2 = graphlayer(g1, 'gl2', 128, nfs, efs, init_scale=np.sqrt(2))
            f_e, f_n = g2[2], g2[3]
            y = [f_e, f_n, b2e, b2n]
            pi = graphblock(y, 'pi', 128, nact, init_scale=np.sqrt(2))

        with tf.variable_scope('value_{}'.format(name), reuse=reuse):
            Y = [X_ns_v, X_nr_v, X_e_v, X_n_v, e2ns_v, e2nr_v, ns2e_v]
            g3 = graphlayer(Y, 'gl3', 128, nfs, efs, init_scale=np.sqrt(2))
            g4 = graphlayer(g3, 'gl4', 128, nfs, efs, init_scale=np.sqrt(2))
            f_e_v, f_n_v = g4[2], g4[3]
            Z = [f_e_v, f_n_v, b2e_v, b2n_v]
            vf = graphblock(Z, 'v', 128, 1, init_scale=np.sqrt(2))

        v0 = vf[:, 0]
        a0 = sample(pi)
        self.initial_state = []  # not stateful

        def step(ob, obs, a_v, *_args, **_kwargs):
            ob_ns, ob_nr, ob_e, ob_n, ob_e2ns, ob_e2nr, ob_ns2e, ob_b2e, ob_b2n = ob
            ob_ns_v, ob_nr_v, ob_e_v, ob_n_v, ob_e2ns_v, ob_e2nr_v, ob_ns2e_v, ob_b2e_v, ob_b2n_v = obs

            a, v = sess.run([a0, v0], {X_ns:ob_ns, X_nr:ob_nr, X_e:ob_e,
                                        X_n:ob_n, e2ns:ob_e2ns, e2nr:ob_e2nr,
                                        ns2e:ob_ns2e, b2e:ob_b2e, b2n:ob_b2n,
                                        X_ns_v:ob_ns_v, X_nr_v:ob_nr_v, X_e_v:ob_e_v,
                                        X_n_v:ob_n_v, e2ns_v:ob_e2ns_v, e2nr_v:ob_e2nr_v,
                                        ns2e_v:ob_ns2e_v, b2e_v:ob_b2e_v, b2n_v:ob_b2n_v})
            return a, v, []  # dummy state

        def value(ob, a_v, *_args, **_kwargs):
            ob_ns_v, ob_nr_v, ob_e_v, ob_n_v, ob_e2ns_v, ob_e2nr_v, ob_ns2e_v, ob_b2e_v, ob_b2n_v = ob
            return sess.run(v0, {X_ns_v:ob_ns_v, X_nr_v:ob_nr_v, X_e_v:ob_e_v,
                                X_n_v:ob_n_v, e2ns_v:ob_e2ns_v, e2nr_v:ob_e2nr_v,
                                ns2e_v:ob_ns2e_v, b2e_v:ob_b2e_v, b2n_v:ob_b2n_v})


        self.X = {"X_ns":X_ns, "X_nr":X_nr, "X_e":X_e, "X_n":X_n, "e2ns":e2ns, "e2nr":e2nr, "ns2e":ns2e, "b2e":b2e, "b2n":b2n}
        self.X_v = {"X_ns":X_ns_v, "X_nr":X_nr_v, "X_e":X_e_v, "X_n":X_n_v, "e2ns":e2ns_v, "e2nr":e2nr_v, "ns2e":ns2e_v, "b2e":b2e_v, "b2n":b2n_v}
        self.A_v = A_v
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value





class CategoricalPolicy(object):
    def __init__(self, sess, ob_space, ac_space, ob_spaces, ac_spaces,
                 nenv, nsteps, nstack, reuse=False, name='model'):
        nbatch = nenv * nsteps
        ob_shape = (nbatch, ob_space.shape[0] * nstack)
        #all_ob_shape = (nbatch, sum([obs.shape[0] for obs in ob_spaces]) * nstack)
        all_ob_shape = ob_shape
        nact = ac_space.n
        all_ac_shape = (nbatch, (sum([ac.n for ac in ac_spaces]) - nact) * nstack)
        X = tf.placeholder(tf.float32, ob_shape)  # obs
        X_v = tf.placeholder(tf.float32, all_ob_shape)
        A_v = tf.placeholder(tf.float32, all_ac_shape)
        with tf.variable_scope('policy_{}'.format(name), reuse=reuse):
            h1 = fc(X, 'fc1', nh=128, init_scale=np.sqrt(2))
            h2 = fc(h1, 'fc2', nh=128, init_scale=np.sqrt(2))
            pi = fc(h2, 'pi', nact, act=lambda x: x)

        with tf.variable_scope('value_{}'.format(name), reuse=reuse):
            #if len(ob_spaces) > 1:
            #    Y = tf.concat([X_v, A_v], axis=1)
            #else:
            Y = X_v
            h3 = fc(Y, 'fc3', nh=256, init_scale=np.sqrt(2))
            h4 = fc(h3, 'fc4', nh=256, init_scale=np.sqrt(2))
            vf = fc(h4, 'v', 1, act=lambda x: x)

        v0 = vf[:, 0]
        a0 = sample(pi)
        self.initial_state = []  # not stateful

        def step(ob, obs, a_v, *_args, **_kwargs):
            if a_v is not None:
                a, v = sess.run([a0, v0], {X: ob, X_v: obs, A_v: a_v})
            else:
                a, v = sess.run([a0, v0], {X: ob, X_v: obs})
            return a, v, []  # dummy state

        def value(ob, a_v, *_args, **_kwargs):
            if a_v is not None:
                return sess.run(v0, {X_v: ob, A_v: a_v})
            else:
                return sess.run(v0, {X_v: ob})

        self.X = X
        self.X_v = X_v
        self.A_v = A_v
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value


class GaussianPolicy(object):
    def __init__(self, sess, ob_space, ac_space, ob_spaces, ac_spaces,
                 nenv, nsteps, nstack, reuse=False, name='model'):
        nbatch = nenv * nsteps
        ob_shape = (nbatch, ob_space.shape[0] * nstack)
        all_ob_shape = (nbatch, sum([obs.shape[0] for obs in ob_spaces]) * nstack)
        nact = ac_space.shape[0]
        all_ac_shape = (nbatch, (sum([ac.shape[0] for ac in ac_spaces]) - nact) * nstack)
        X = tf.placeholder(tf.float32, ob_shape)  # obs
        X_v = tf.placeholder(tf.float32, all_ob_shape)
        A_v = tf.placeholder(tf.float32, all_ac_shape)
        with tf.variable_scope('policy_{}'.format(name), reuse=reuse):
            h1 = fc(X, 'fc1', nh=64, init_scale=np.sqrt(2), act=tf.nn.tanh)
            h2 = fc(h1, 'fc2', nh=64, init_scale=np.sqrt(2), act=tf.nn.tanh)
            pi = fc(h2, 'pi', nact, act=lambda x: x, init_scale=0.01)

        with tf.variable_scope('policy_{}'.format(name), reuse=reuse):
            logstd = tf.get_variable('sigma', shape=[nact], dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.0))
            logstd = tf.expand_dims(logstd, 0)
            std = tf.exp(logstd)
            std = tf.tile(std, [nbatch, 1])

        with tf.variable_scope('value_{}'.format(name), reuse=reuse):
            if len(ob_spaces) > 1:
                Y = tf.concat([X_v, A_v], axis=1)
            else:
                Y = X_v
            h3 = fc(Y, 'fc3', nh=64, init_scale=np.sqrt(2), act=tf.nn.tanh)
            h4 = fc(h3, 'fc4', nh=64, init_scale=np.sqrt(2), act=tf.nn.tanh)
            vf = fc(h4, 'v', 1, act=lambda x: x)

        v0 = vf[:, 0]
        a0 = pi + tf.random_normal(tf.shape(std), 0.0, 1.0) * std

        self.initial_state = []  # not stateful

        def step(ob, obs, a_v, *_args, **_kwargs):
            if a_v is not None:
                a, v = sess.run([a0, v0], {X: ob, X_v: obs, A_v: a_v})
            else:
                a, v = sess.run([a0, v0], {X: ob, X_v: obs})
            return a, v, []  # dummy state

        def value(ob, a_v, *_args, **_kwargs):
            if a_v is not None:
                return sess.run(v0, {X_v: ob, A_v: a_v})
            else:
                return sess.run(v0, {X_v: ob})

        self.X = X
        self.X_v = X_v
        self.A_v = A_v
        self.pi = pi
        self.vf = vf
        self.std = std
        self.logstd = logstd
        self.step = step
        self.value = value
        self.mean_std = tf.concat([pi, std], axis=1)


class MultiCategoricalPolicy(object):
    def __init__(self, sess, ob_space, ac_space, ob_spaces, ac_spaces,
                 nenv, nsteps, nstack, reuse=False, name='model'):
        nbins = 11
        nbatch = nenv * nsteps
        ob_shape = (nbatch, ob_space.shape[0] * nstack)
        all_ob_shape = (nbatch, sum([obs.shape[0] for obs in ob_spaces]) * nstack)
        nact = ac_space.shape[0]
        all_ac_shape = (nbatch, (sum([ac.shape[0] for ac in ac_spaces]) - nact) * nstack)
        X = tf.placeholder(tf.float32, ob_shape)  # obs
        X_v = tf.placeholder(tf.float32, all_ob_shape)
        A_v = tf.placeholder(tf.float32, all_ac_shape)
        with tf.variable_scope('policy_{}'.format(name), reuse=reuse):
            h1 = fc(X, 'fc1', nh=128, init_scale=np.sqrt(2))
            h2 = fc(h1, 'fc2', nh=128, init_scale=np.sqrt(2))
            pi = fc(h2, 'pi', nact * nbins, act=lambda x: x)

        with tf.variable_scope('value_{}'.format(name), reuse=reuse):
            if len(ob_spaces) > 1:
                Y = tf.concat([X_v, A_v], axis=1)
            else:
                Y = X_v
            h3 = fc(Y, 'fc3', nh=256, init_scale=np.sqrt(2))
            h4 = fc(h3, 'fc4', nh=256, init_scale=np.sqrt(2))
            vf = fc(h4, 'v', 1, act=lambda x: x)

        v0 = vf[:, 0]
        pi = tf.reshape(pi, [nbatch, nact, nbins])
        a0 = sample(pi, axis=2)
        self.initial_state = []  # not stateful

        def step(ob, obs, a_v, *_args, **_kwargs):
            # output continuous actions within [-1, 1]
            if a_v is not None:
                a, v = sess.run([a0, v0], {X: ob, X_v: obs, A_v: a_v})
            else:
                a, v = sess.run([a0, v0], {X: ob, X_v: obs})
            a = transform(a)
            return a, v, []  # dummy state

        def value(ob, a_v, *_args, **_kwargs):
            if a_v is not None:
                return sess.run(v0, {X_v: ob, A_v: a_v})
            else:
                return sess.run(v0, {X_v: ob})

        def transform(a):
            # transform from [0, 9] to [-0.8, 0.8]
            a = np.array(a, dtype=np.float32)
            a = (a - (nbins - 1) / 2) / (nbins - 1) * 2.0
            return a

        self.X = X
        self.X_v = X_v
        self.A_v = A_v
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

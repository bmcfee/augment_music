import optimus
import numpy as np

GRAPH_NAME = 'multiregressor'
NUM_FREQ_COEFFS = 216


def uniform_init(param, bias=0.0, scale=1e-2):
    '''Uniform random initialization'''

    param.value = bias + scale * np.random.uniform(low=-1.0,
                                                   high=1.0,
                                                   size=param.shape)


def param_init(nodes, scale=0.01, skip_biases=True):
    for n in nodes:
        for k, p in n.params.items():
            if 'bias' in k and skip_biases:
                continue
            # optimus.random_init(p, 0, scale)
            uniform_init(p, bias=0, scale=scale)


def beastly_network(num_frames, num_classes, size='large'):
    # Kernel counts, per layer
    k0, k1, k2 = dict(
        small=(10, 20, 48),
        med=(12, 24, 256),
        large=(24, 48, 96),
        xlarge=(20, 40, 128),
        xxlarge=(24, 48, 256))[size]

    # Input dimensions
    # 44 frames is approx 1 second
    n0, n1 = {
        1: (1, 1),
        4: (3, 2),
        8: (5, 3),
        10: (3, 3),
        20: (5, 5),
        44: (9, 7),
        128: (9, 7)}[num_frames]

    # Pool shapes
    p0, p1 = {
        1: (1, 1),
        4: (1, 1),
        8: (1, 1),
        10: (2, 2),
        12: (2, 2),
        20: (2, 2),
        44: (2, 2),
        128: (4, 1)}[num_frames]

    input_data = optimus.Input(
        name='X',
        shape=(None, 1, num_frames, NUM_FREQ_COEFFS))

    class_targets = optimus.Input(
        name='Y',
        shape=(None, num_classes))

    learning_rate = optimus.Input(
        name='learning_rate',
        shape=None)

    weight_decay = optimus.Input(
        name='weight_decay',
        shape=None)

    dropout = optimus.Input(
        name='dropout',
        shape=None)

    inputs = [input_data, class_targets, learning_rate]

    # 1.2 Create Nodes
    layer0 = optimus.Conv3D(
        name='layer0',
        input_shape=input_data.shape,
        weight_shape=(k0, None, n0, 13),
        pool_shape=(p0, 3),
        act_type='relu')

    layer1 = optimus.Conv3D(
        name='layer1',
        input_shape=layer0.output.shape,
        weight_shape=(k1, None, n1, 9),
        pool_shape=(p1, 1),
        act_type='relu')

    max_pool = optimus.Max(
        name='max_pool',
        axis=2)

    # TODO(ejhumphrey): optimus.Max doesn't propagate shapes...
    mp_output_shape = list(layer1.output.shape)
    mp_output_shape[-2] = 1
    layer2 = optimus.Affine(
        name='layer2',
        input_shape=layer1.output.shape,
        output_shape=(None, k2),
        act_type='relu')

    l2_penalty = optimus.WeightDecayPenalty('l2_penalty')
    classifier = optimus.Affine(
        name='classifier',
        input_shape=layer2.output.shape,
        output_shape=(None, num_classes),
        act_type='sigmoid')

    param_nodes = [layer0, layer1, layer2, classifier]

    # 1.2 Create Loss
    # ---------------
    xentropy = optimus.CrossEntropyLoss(name='cross_entropy')
    total_loss = optimus.Add('total_loss', num_inputs=2)

    # Graph outputs
    loss = optimus.Output(name='loss')
    Z0 = optimus.Output(name='Z0')
    Z1 = optimus.Output(name='Z1')
    Z2 = optimus.Output(name='Z2')
    # Z3 = optimus.Output(name='Z3')
    prediction = optimus.Output(name='Z')

    # Dropout
    layer2.enable_dropout()

    # 2. Define Edges
    base_edges = [
        (input_data, layer0.input),
        (layer0.output, layer1.input),
        (layer0.output, Z0),
        (layer1.output, layer2.input),
        (layer1.output, Z1),
        (layer2.output, classifier.input),
        (layer2.output, Z2),
        (classifier.output, prediction)]

    trainer_edges = base_edges + [(classifier.output, xentropy.prediction),
                                  (class_targets, xentropy.target),
                                  (xentropy.output, total_loss.input_0),
                                  (classifier.weights, l2_penalty.input),
                                  (weight_decay, l2_penalty.weight),
                                  (l2_penalty.output, total_loss.input_1),
                                  (dropout, layer2.dropout),
                                  (total_loss.output, loss)]

    updates = optimus.ConnectionManager(
        map(lambda n: (learning_rate, n.weights), param_nodes) +
        map(lambda n: (learning_rate, n.bias), param_nodes))

    param_init(param_nodes, scale=0.01, skip_biases=False)

    trainer = optimus.Graph(
        name=GRAPH_NAME,
        inputs=inputs + [weight_decay, dropout],
        nodes=param_nodes + [max_pool, xentropy, l2_penalty, total_loss],
        connections=optimus.ConnectionManager(trainer_edges).connections,
        outputs=[loss, prediction, Z0, Z1, Z2],
        loss=loss,
        updates=updates.connections,
        verbose=True)

    layer2.disable_dropout()

    predictor = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data],
        nodes=param_nodes + [max_pool],
        connections=optimus.ConnectionManager(base_edges).connections,
        outputs=[prediction, Z0, Z1, Z2],
        verbose=True)

    return trainer, predictor

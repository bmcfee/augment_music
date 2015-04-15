import optimus

GRAPH_NAME = 'multiregressor'
NUM_FREQ_COEFFS = 216


def param_init(nodes, skip_biases=True):
    for n in nodes:
        for k, p in n.params.items():
            if 'bias' in k and skip_biases:
                continue
            optimus.random_init(p, 0, 0.01)


def beastly_network(num_frames, num_classes, size='large'):
    # Kernel counts, per layer
    k0, k1, k2, k3 = dict(
        small=(10, 20, 40, 96),
        med=(12, 24, 48, 128),
        large=(16, 32, 64, 192),
        xlarge=(20, 40, 80, 256),
        xxlarge=(24, 48, 96, 512))[size]

    # Input dimensions
    # 44 frames is approx 1 second
    n0, n1, n2 = {
        1: (1, 1, 1),
        4: (3, 2, 1),
        8: (5, 3, 2),
        10: (3, 3, 1),
        20: (5, 5, 1),
        44: (9, 9, 5)}[num_frames]

    # Pool shapes
    p0, p1, p2 = {
        1: (1, 1, 1),
        4: (1, 1, 1),
        8: (1, 1, 1),
        10: (2, 2, 1),
        12: (2, 2, 1),
        20: (2, 2, 2),
        44: (2, 2, 1)}[num_frames]

    input_data = optimus.Input(
        name='cqt',
        shape=(None, 1, num_frames, NUM_FREQ_COEFFS))

    class_targets = optimus.Input(
        name='class_targets',
        shape=(None, num_classes))

    learning_rate = optimus.Input(
        name='learning_rate',
        shape=None)

    inputs = [input_data, class_targets, learning_rate]

    # 1.2 Create Nodes
    layer0 = optimus.Conv3D(
        name='layer0',
        input_shape=input_data.shape,
        weight_shape=(k0, None, n0, 17),
        pool_shape=(p0, 2),
        act_type='relu')

    layer1 = optimus.Conv3D(
        name='layer1',
        input_shape=layer0.output.shape,
        weight_shape=(k1, None, n1, 17),
        pool_shape=(p1, 2),
        act_type='relu')

    layer2 = optimus.Conv3D(
        name='layer2',
        input_shape=layer1.output.shape,
        weight_shape=(k2, None, n2, 13),
        pool_shape=(p2, 1),
        act_type='relu')

    layer3 = optimus.Affine(
        name='layer3',
        input_shape=layer2.output.shape,
        output_shape=(None, k3),
        act_type='relu')

    classifier = optimus.Affine(
        name='classifier',
        input_shape=layer3.output.shape,
        output_shape=(None, num_classes),
        act_type='sigmoid')

    param_nodes = [layer0, layer1, layer2, layer3, classifier]

    # 1.2 Create Loss
    # ---------------
    xentropy = optimus.CrossEntropyLoss(name='cross_entropy')

    # Graph outputs
    loss = optimus.Output(name='loss')
    prediction = optimus.Output(name='prediction')

    # 2. Define Edges
    base_edges = [
        (input_data, layer0.input),
        (layer0.output, layer1.input),
        (layer1.output, layer2.input),
        (layer2.output, layer3.input),
        (layer3.output, classifier.input),
        (classifier.output, prediction)]

    trainer_edges = base_edges + [(classifier.output, xentropy.prediction),
                                  (class_targets, xentropy.target),
                                  (xentropy.output, loss)]

    updates = optimus.ConnectionManager(
        map(lambda n: (learning_rate, n.weights), param_nodes) +
        map(lambda n: (learning_rate, n.bias), param_nodes))

    param_init(param_nodes)

    trainer = optimus.Graph(
        name=GRAPH_NAME,
        inputs=inputs,
        nodes=param_nodes + [xentropy],
        connections=optimus.ConnectionManager(trainer_edges).connections,
        outputs=[loss, prediction],
        loss=loss,
        updates=updates.connections,
        verbose=True)

    predictor = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data],
        nodes=param_nodes,
        connections=optimus.ConnectionManager(base_edges).connections,
        outputs=[prediction],
        verbose=True)

    return trainer, predictor

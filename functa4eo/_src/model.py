import keras
from knerf._src.layers.siren import SirenLayer


def init_siren_model(omega_0: float = 30.0, omega: float = 1.0, c: float = 6.0):
    spatial_input = keras.Input(shape=(2,), name="spatial")
    temporal_input = keras.Input(shape=(1,), name="temporal")
    x_spacetime = keras.layers.concatenate(
        [
            temporal_input,
            spatial_input,
        ]
    )
    x_spacetime = SirenLayer(units=256, omega=omega_0, c=c, layer_type="first")(
        x_spacetime
    )
    x_spacetime = SirenLayer(units=256, omega=omega, c=c)(x_spacetime)
    x_spacetime = SirenLayer(units=256, omega=omega, c=c)(x_spacetime)
    x_spacetime = SirenLayer(units=256, omega=omega, c=c)(x_spacetime)
    x_spacetime = SirenLayer(units=256, omega=omega, c=c)(x_spacetime)
    model_output = SirenLayer(units=1, omega=omega, c=c, layer_type="last")(x_spacetime)

    model = keras.Model(
        inputs=[temporal_input, spatial_input], outputs=model_output, name="siren"
    )

    return model


def init_siren_multihead_model(
    omega_0: float = 30.0, omega: float = 1.0, c: float = 6.0
):
    # spatial head
    spatial_input = keras.Input(shape=(2,), name="spatial")
    x_space = SirenLayer(units=128, omega=omega_0, c=c, layer_type="first")(
        spatial_input
    )
    x_space = SirenLayer(units=128, omega=omega, c=c)(x_space)
    x_space = SirenLayer(units=128, omega=omega, c=c)(x_space)
    spatial_output = SirenLayer(units=128, omega=omega)(x_space)

    # temporal head
    temporal_input = keras.Input(shape=(1,), name="temporal")
    x_time = SirenLayer(units=128, omega=omega_0, c=c, layer_type="first")(
        temporal_input
    )
    temporal_output = SirenLayer(units=128, omega=omega)(x_time)

    # spatial_encoder = keras.Model(spatial_input, spatial_output, name="spatial_encoder")
    # temporal_encoder = keras.Model(temporal_input, temporal_output, name="temporal_encoder")

    # together
    x_spacetime = keras.layers.add([spatial_output, temporal_output])
    x_spacetime = SirenLayer(units=128, omega=omega, c=c)(x_spacetime)
    x_spacetime = SirenLayer(units=128, omega=omega, c=c)(x_spacetime)
    model_output = SirenLayer(units=1, omega=omega, c=c, layer_type="last")(x_spacetime)

    model = keras.Model(
        inputs=[temporal_input, spatial_input], outputs=model_output, name="multi_head"
    )

    return model

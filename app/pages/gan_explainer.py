from __future__ import annotations

import streamlit as st

from app.styles import hero


def render() -> None:
    hero("How GAN Works", "Learn the loop behind every generated concept.", "Learning lab")
    topic = st.radio("Choose a topic", ["Overview", "Generator", "Discriminator", "Training loop"], horizontal=True)
    if topic == "Overview":
        st.markdown(
            """
            ### What is a Generative Adversarial Network (GAN)?

            A **Generative Adversarial Network (GAN)** is a deep learning architecture
            consisting of two neural networks: the **Generator** and the
            **Discriminator**.

            These networks compete against each other to generate realistic synthetic
            images. The Generator creates artificial images, while the Discriminator
            determines whether an image is real (from the dataset) or fake (generated
            by the Generator).

            GAN was introduced by Ian Goodfellow and his colleagues in 2014.

            ---

            ### 1. Understanding the Two Main Components

            #### A. Generator (G) – The Image Creator

            The Generator is responsible for creating new synthetic fashion images.

            **How does it work?**

            1. Receives a random noise vector containing 100 numerical values.
            2. Processes the noise through multiple neural network layers.
            3. Learns visual patterns such as clothing shapes, edges, textures,
              and structures from the training dataset.
            4. Converts the learned features into 784 pixel values.
            5. Reshapes these values into a 28 × 28 grayscale image.
            6. Uses the Tanh activation function to keep output values between -1 and +1.

            **Example:**

            Random Noise → Generator → Synthetic Shirt / Shoe / Bag Image

            The Generator does not simply retrieve or copy an existing image.
            It learns patterns from training examples and uses them to create
            new synthetic images.

            #### B. Discriminator (D) – The Image Judge

            The Discriminator works as a binary classifier.

            Its responsibility is to determine whether an image is:

            - **Real:** An actual image taken from the Fashion-MNIST dataset.
            - **Fake:** An artificially generated image produced by the Generator.

            It receives both real and fake images and produces a numerical logit
            representing its prediction.

            **Example:**

            | Input Image | Discriminator Prediction |
            |---|---|
            | Real Fashion Image | Positive logit |
            | Generated Fashion Image | Negative logit |

            The Discriminator learns to identify differences between real training
            images and synthetic images.

            **Important:** The Discriminator is not a clothing classifier.
            It identifies real versus fake images, not whether an image is a
            shirt, shoe, dress, or bag.

            ---

            ### 2. How GAN Learns from the Dataset

            Our project uses the **Fashion-MNIST dataset**, which contains
            70,000 grayscale clothing images.

            It consists of 10 fashion categories:

            1. T-shirt / Top
            2. Trouser
            3. Pullover
            4. Dress
            5. Coat
            6. Sandal
            7. Shirt
            8. Sneaker
            9. Bag
            10. Ankle Boot

            The dataset is divided into:
            - 60,000 training images.
            - 10,000 testing images.

            During training, the Generator and Discriminator learn from the
            visual patterns present in these images.

            **What does the GAN learn?**

            - Clothing shapes and silhouettes.
            - Edges and boundaries.
            - Pixel arrangements.
            - Patterns and visual structures.
            - Common characteristics of different clothing items.

            Before training, images undergo preprocessing:

            ```text
            Original Fashion Images
                      |
                      v
                Image Resizing
                      |
                      v
                Normalization
                      |
                      v
                Tensor Conversion
                      |
                      v
                Training Dataset
            ```

            All images are represented as 1 × 28 × 28 grayscale tensors.
            Pixel values are normalized from [0, 1] to [-1, 1] to match
            the Generator's Tanh output.

            ---

            ### 3. Complete GAN Architecture

            The following flowchart explains how the Generator and Discriminator
            work together.

            ```text
                        RANDOM NOISE
                          (100 Values)
                              |
                              v
                        +-------------+
                        |  GENERATOR  |
                        |     (G)     |
                        +-------------+
                              |
                              v
                        FAKE FASHION IMAGE
                              |
                              |
            REAL FASHION       |
            IMAGE FROM         |
            DATASET            |
                |             |
                v             v
              +-------------------+
              |                   |
              |   DISCRIMINATOR   |
              |        (D)        |
              |                   |
              +-------------------+
                        |
                        v
                  REAL OR FAKE?
                        |
                        v
                    LOSS FUNCTION
                        |
                        v
                  BACKPROPAGATION
                        |
                        v
                  UPDATE WEIGHTS
                        |
                        v
                  IMPROVE BOTH
                  NETWORKS
                        |
                        v
                  REPEAT TRAINING
            ```

            ---

            ### 4. How Does the Generator Actually Create an Image?

            The Generator starts with a 100-dimensional random noise vector.

            For the Basic GAN, the architecture is:

            ```text
            Random Noise
                |
                v
            Fully Connected Layer
                |
                v
            100 → 256
                |
                v
            256 → 512
                |
                v
            512 → 1024
                |
                v
            1024 → 784
                |
                v
            Tanh Activation
                |
                v
            Reshape to 1 × 28 × 28
                |
                v
            Generated Fashion Image
            ```

            The network progressively transforms random numerical values into
            learned image features.

            The project also includes DCGAN, which uses convolutional and
            transposed convolutional layers to learn spatial image structures.

            ---

            ### 5. How Does the Discriminator Judge an Image?

            The Discriminator receives an image and extracts visual features.

            It learns to recognize:

            - Low-level features: edges, strokes, and contrast.
            - Mid-level features: shapes and clothing silhouettes.
            - Authenticity cues: patterns that distinguish real images
              from generated images.

            Its output is a single logit.

            ```text
            Input Image (1 × 28 × 28)
                      |
                      v
                Feature Extraction
                      |
                      v
                Neural Network
                      |
                      v
                Real/Fake Logit
            ```

            The project uses BCEWithLogitsLoss to calculate the
            Discriminator's and Generator's training losses.

            ---

            ### 6. The Adversarial Training Process

            GAN training works through competition between its two networks.

            **Step 1: Train the Discriminator**

            - Load real images from Fashion-MNIST.
            - Generate fake images using random noise.
            - Give real images the target label 1.
            - Give fake images the target label 0.
            - Calculate the Discriminator loss.
            - Update Discriminator weights.

            During this step, fake images are detached from the computation
            graph so the Generator is not updated.

            **Step 2: Train the Generator**

            - Generate a fresh random noise batch.
            - Pass it through the Generator.
            - Send generated images to the Discriminator.
            - Set target labels to real (1).
            - Calculate Generator loss.
            - Update Generator weights through backpropagation.

            The Generator tries to fool the Discriminator into classifying
            its synthetic images as real.

            **Step 3: Repeat**

            These steps repeat over multiple epochs until the Generator
            learns to produce increasingly realistic fashion images.

            ```text
                    START TRAINING
                          |
                          v
                    LOAD REAL IMAGES
                          |
                          v
                    SAMPLE NOISE
                          |
                          v
                    GENERATE FAKE
                          |
                          v
                  TRAIN DISCRIMINATOR
                          |
                          v
                    CALCULATE D LOSS
                          |
                          v
                    UPDATE D WEIGHTS
                          |
                          v
                    SAMPLE NEW NOISE
                          |
                          v
                    GENERATE NEW FAKE
                          |
                          v
                    TRAIN GENERATOR
                          |
                          v
                    CALCULATE G LOSS
                          |
                          v
                    UPDATE G WEIGHTS
                          |
                          v
                    SAVE METRICS
                          |
                          v
                    NEXT EPOCH
            ```

            ---

            ### 7. What Happens After Training?

            Once training is complete, the Generator learns to produce
            synthetic fashion images.

            We save the trained Generator weights in a checkpoint file.

            For example:

            `generator.pth`

            During normal application usage, we do not need to train the GAN
            again or load the Discriminator.

            We simply:

            1. Load the trained Generator.
            2. Generate a new random noise vector.
            3. Pass it through the Generator.
            4. Receive a synthetic fashion image.
            5. Convert the image into displayable pixel values.
            6. Display the generated image in Fashion Studio.

            ```text
                    USER OPENS APP
                          |
                          v
                    LOAD GENERATOR
                          |
                          v
                    RANDOM NOISE
                          |
                          v
                    TRAINED GENERATOR
                          |
                          v
                    GENERATED IMAGE
                          |
                          v
                    DISPLAY IN STREAMLIT
                          |
                          v
                    DOWNLOAD IMAGE
            ```

            ---

            ### 8. Basic GAN vs DCGAN

            This project implements two GAN architectures.

            | Feature | Basic GAN | DCGAN |
            |---|---|---|
            | Architecture | Fully Connected | Convolutional |
            | Generator | Linear Layers | Transposed Convolutions |
            | Discriminator | Linear Layers | Convolutional Layers |
            | Image Processing | Flattened Pixels | Spatial Feature Maps |
            | Spatial Information | Less Explicit | Better Preserved |
            | Main Purpose | Basic GAN Learning | Image Generation |

            Both architectures use the same adversarial learning principle.
            The main difference is how they process and generate images.

            ---

            ### 9. Important Concepts to Remember

            | Concept | Meaning |
            |---|---|
            | GAN | Generative Adversarial Network |
            | Generator | Creates synthetic images |
            | Discriminator | Distinguishes real from fake |
            | Latent Vector | Random noise input |
            | Dataset | Provides real training images |
            | Epoch | One complete pass through training data |
            | Batch Size | Images processed together |
            | Backpropagation | Calculates gradients |
            | Optimizer | Updates network weights |
            | Loss Function | Measures training objectives |
            | Tanh | Restricts output between -1 and +1 |
            | DCGAN | Deep Convolutional GAN |

            **The main idea:**

            The Generator learns to create realistic images, while the
            Discriminator learns to detect generated images. Their competition
            gradually improves the quality of synthetic images.

            This is the fundamental working principle behind GAN FashionVerse.
            """
        )

        st.success(
            "Key takeaway: Generator creates, Discriminator judges, "
            "and adversarial training improves image generation."
        )
    elif topic == "Generator":
        st.subheader("The Generator: Creating New Fashion Images")

        st.markdown(
            """
            ### 1. What is a Generator?

            The **Generator (G)** is one of the two main components of a
            Generative Adversarial Network (GAN).

            Its primary responsibility is to create new synthetic images
            that look similar to real images from the training dataset.

            In GAN FashionVerse, the Generator learns to create fashion
            images such as shirts, trousers, dresses, shoes, and bags
            using patterns learned from the Fashion-MNIST dataset.

            Unlike a traditional image retrieval system, the Generator
            does not simply select an existing image from a database.
            It learns a mathematical transformation that converts
            random noise into a new synthetic image.

            **Simple Example:**

            Random Noise → Generator → New Fashion Image

            ---

            ### 2. How Does the Generator Work?

            The Generator follows a sequence of operations.

            #### Step 1: Random Noise (Latent Vector)

            The Generator receives a random vector called a
            **Latent Vector (Z)** as input.

            In our project, this vector contains 100 numerical values.

            Example:

            ```text
            Z = [
                0.45, -0.21, 0.89, 0.12,
                -0.56, 0.78, 0.34, ...
            ]
            ```

            These values are randomly sampled from a normal distribution.

            Each vector represents a different starting point for image
            generation.

            Different noise vectors can produce different fashion images.

            **Why do we use random noise?**

            - To introduce variation in generated images.
            - To generate multiple different outputs.
            - To allow the Generator to learn different visual patterns.
            - To explore different points in the learned latent space.

            #### Step 2: Fully Connected Layers

            The random noise is passed through multiple fully connected
            (Linear) layers.

            Each layer transforms the input into a more useful
            learned feature representation.

            Our Basic GAN uses the following architecture:

            ```text
            Input Noise: 100 Values
                    |
                    v
              Linear Layer
                  100 → 256
                    |
                    v
                  ReLU
                    |
                    v
              Linear Layer
                  256 → 512
                    |
                    v
                  ReLU
                    |
                    v
              Linear Layer
                  512 → 1024
                    |
                    v
                  ReLU
                    |
                    v
              Linear Layer
                  1024 → 784
                    |
                    v
                Tanh
                    |
                    v
              Output: 784 Values
            ```

            The number of neurons increases as information moves
            through the network.

            This allows the Generator to learn increasingly complex
            relationships between the random input and image pixels.

            #### Step 3: Activation Functions

            Activation functions introduce non-linearity into the
            neural network.

            Our Generator uses two important activation functions.

            **A. ReLU (Rectified Linear Unit)**

            ReLU is used in hidden layers.

            Mathematical formula:

            f(x) = max(0, x)

            It converts negative values into zero while keeping
            positive values unchanged.

            Example:

            | Input | ReLU Output |
            |---|---|
            | -5 | 0 |
            | -2 | 0 |
            | 0 | 0 |
            | 3 | 3 |
            | 7 | 7 |

            **Purpose:** Helps the Generator learn complex
            non-linear relationships.

            **B. Tanh (Hyperbolic Tangent)**

            Tanh is used in the final output layer.

            Its output range is:

            -1 to +1

            Why do we use Tanh?

            Because our Fashion-MNIST images are normalized to
            the same range during preprocessing.

            This ensures that generated pixel values have a
            compatible range with the real training images.

            #### Step 4: Image Formation

            The final Linear layer produces 784 numerical values.

            Since our Fashion-MNIST images have dimensions of
            28 × 28 pixels:

            28 × 28 = 784

            These 784 values are reshaped into a grayscale image.

            ```text
            784 Numerical Values
                    |
                    v
                Reshape
                    |
                    v
              1 × 28 × 28 Tensor
                    |
                    v
              Generated Fashion Image
            ```

            The output is a synthetic image that can be displayed
            in the Fashion Studio.

            ---

            ### 3. Complete Generator Architecture

            The complete process can be represented as:

            ```text
                      RANDOM NOISE
                        (100 Values)
                            |
                            v
                      FULLY CONNECTED
                        100 → 256
                            |
                            v
                          ReLU
                            |
                            v
                      FULLY CONNECTED
                        256 → 512
                            |
                            v
                          ReLU
                            |
                            v
                      FULLY CONNECTED
                      512 → 1024
                            |
                            v
                          ReLU
                            |
                            v
                      FULLY CONNECTED
                      1024 → 784
                            |
                            v
                          Tanh
                            |
                            v
                      RESHAPE IMAGE
                        1 × 28 × 28
                            |
                            v
                      GENERATED IMAGE
            ```

            ---

            ### 4. How Does the Generator Learn?

            Initially, the Generator has randomly initialized weights.
            Therefore, its generated images may look like random noise.

            The Generator improves through feedback from the Discriminator.

            **Training process:**

            1. Generate a fake fashion image from random noise.
            2. Send the generated image to the Discriminator.
            3. The Discriminator evaluates whether it looks real or fake.
            4. Calculate the Generator loss.
            5. Use backpropagation to calculate gradients.
            6. Update the Generator's weights using an optimizer.
            7. Repeat the process over multiple training iterations.

            The Generator tries to make the Discriminator classify
            its generated images as real.

            **Important:** During the Generator update, gradients
            pass through the Discriminator but only the Generator's
            weights are updated.

            ```text
                  RANDOM NOISE
                        |
                        v
                    GENERATOR
                        |
                        v
                    FAKE IMAGE
                        |
                        v
                  DISCRIMINATOR
                        |
                        v
                  GENERATOR LOSS
                        |
                        v
                  BACKPROPAGATION
                        |
                        v
                  UPDATE G WEIGHTS
                        |
                        v
                IMPROVED GENERATOR
            ```

            ---

            ### 5. Generator Loss Function

            The Generator tries to fool the Discriminator.

            Its objective is to make generated images appear real.

            The commonly used non-saturating Generator loss is:

            L(G) = -E[log(D(G(z)))]

            Where:

            - G = Generator
            - D = Discriminator
            - z = Random noise vector
            - D(G(z)) = Discriminator's estimated probability
              that the generated image is real.

            The Generator attempts to increase D(G(z)).

            A lower Generator loss generally indicates that the
            Discriminator is assigning higher real scores to generated
            images, although loss alone does not determine image quality.

            ---

            ### 6. Basic GAN Generator vs DCGAN Generator

            Our project implements two types of Generator architectures.

            **A. Basic GAN Generator**

            - Uses fully connected Linear layers.
            - Converts a 100-dimensional noise vector into 784 pixels.
            - Produces a 28 × 28 grayscale image.
            - Flattens the image representation during processing.

            Architecture:

            ```text
            Noise (100)
                |
                v
            Linear Layers
                |
                v
            ReLU Activations
                |
                v
            Linear (1024 → 784)
                |
                v
            Tanh
                |
                v
            Image (28 × 28)
            ```

            **B. DCGAN Generator**

            DCGAN stands for Deep Convolutional Generative Adversarial Network.

            Unlike Basic GAN, it uses transposed convolutional layers
            to increase the spatial dimensions of feature maps.

            These layers progressively construct an image while
            preserving spatial relationships.

            Example conceptual flow:

            ```text
            Random Noise
                |
                v
            Initial Feature Map
                |
                v
            Transposed Convolution
                |
                v
            Larger Feature Maps
                |
                v
            More Upsampling Layers
                |
                v
            Final Tanh
                |
                v
            Generated Image
            ```

            DCGAN is particularly useful for image generation because
            convolutional architectures can learn spatial patterns
            more effectively than simply treating every pixel as
            an independent feature.

            ---

            ### 7. What Happens When a User Clicks Generate?

            After training, we save the Generator's learned weights
            in a checkpoint file.

            For example:

            `generator.pth`

            When the user opens Fashion Studio and clicks
            **Generate Fashion Image**, the following happens:

            1. The trained Generator model is loaded.
            2. A new random noise vector is created.
            3. The noise is passed through the Generator.
            4. The Generator produces 784 pixel values.
            5. The output is reshaped into a 28 × 28 image.
            6. Pixel values are converted back into a displayable range.
            7. Streamlit displays the generated image.

            ```text
                    USER CLICKS GENERATE
                              |
                              v
                      LOAD GENERATOR
                              |
                              v
                        RANDOM NOISE
                              |
                              v
                        GENERATOR
                              |
                              v
                      GENERATED IMAGE
                              |
                              v
                      DISPLAY IN UI
                              |
                              v
                        DOWNLOAD IMAGE
            ```

            **Important:** During image generation, the Discriminator
            is not required. Only the trained Generator is needed.

            ---

            ### 8. Important Generator Parameters

            | Parameter | Value | Purpose |
            |---|---|---|
            | Latent Dimension | 100 | Size of random noise vector |
            | Input Size | 100 | Generator input features |
            | Hidden Layers | 256, 512, 1024 | Learn image representations |
            | Output Size | 784 | Number of image pixels |
            | Image Height | 28 | Output image height |
            | Image Width | 28 | Output image width |
            | Channels | 1 | Grayscale image |
            | Hidden Activation | ReLU | Introduces non-linearity |
            | Output Activation | Tanh | Normalizes output |
            | Optimizer | Adam | Updates network weights |

            ---

            ### 9. Important Concepts – Quick Revision

            | Term | Explanation |
            |---|---|
            | Generator | Creates synthetic images |
            | Latent Vector | Random numerical input |
            | Latent Dimension | Number of values in the noise vector |
            | Linear Layer | Fully connected neural network layer |
            | ReLU | Hidden-layer activation function |
            | Tanh | Final activation function |
            | Feature Map | Learned representation of image features |
            | Transposed Convolution | Increases spatial dimensions |
            | Generator Loss | Measures how well G fools D |
            | Backpropagation | Calculates gradients |
            | Adam | Optimizer used for weight updates |

            **Final Takeaway:**

            The Generator is the creative part of GAN FashionVerse.
            It takes random noise as input, transforms it through
            multiple neural network layers, and produces a new
            synthetic fashion image.

            During training, feedback from the Discriminator helps
            the Generator continuously improve its image-generation
            capabilities.
            """
        )

        st.success(
            "Key takeaway: Random Noise → Generator → Neural Network "
            "Layers → Tanh → Synthetic Fashion Image."
        )

        st.info(
            "In FashionVerse, the Generator learns fashion patterns "
            "from Fashion-MNIST. It generates new 28 × 28 grayscale "
            "images rather than retrieving existing dataset images."
        )

        with st.expander("Explore the Generator's random input"):
            latent_dim = 100

            st.write(
                "The Generator receives a 100-dimensional random "
                "latent vector as input."
            )

            import torch

            sample_noise = torch.randn(1, latent_dim)

            st.write("Sample latent vector (first 10 values):")
            st.code(
                str(sample_noise[0][:10].tolist()),
                language="python"
            )

            st.caption(
                "Every time a new noise vector is sampled, the "
                "Generator gets a different starting point for "
                "creating an image."
            )
    elif topic == "Discriminator":
        st.subheader("The Discriminator: Judging Image Authenticity")

        st.markdown(
            """
            ### 1. What is a Discriminator?

            The **Discriminator (D)** is the second major component of a
            Generative Adversarial Network (GAN).

            Its primary responsibility is to determine whether an input
            image is:

            - **Real:** An actual image taken from the training dataset.
            - **Fake:** An artificially generated image created by the Generator.

            The Discriminator works like a judge or detective. It examines
            images and learns to distinguish real fashion images from
            artificially generated ones.

            In GAN FashionVerse, the Discriminator receives images from
            two sources:

            1. Real clothing images from the Fashion-MNIST dataset.
            2. Synthetic clothing images created by the Generator.

            Its objective is to correctly identify both types of images.

            **Simple Example:**

            Real Fashion Image → Discriminator → REAL

            Generated Fashion Image → Discriminator → FAKE

            ---

            ### 2. How Does the Discriminator Work?

            The Discriminator follows a sequence of operations.

            #### Step 1: Receive Input Images

            The Discriminator receives two types of images.

            **A. Real Images**

            These images come directly from the Fashion-MNIST dataset.

            Examples:
            - Real shirts.
            - Real trousers.
            - Real dresses.
            - Real shoes.
            - Real bags.

            **B. Fake Images**

            These images are generated by the Generator using random noise.

            Initially, generated images may look like random pixels.
            As training progresses, they become more similar to the
            real images in the training dataset.

            ```text
                    REAL IMAGE
                FROM FASHION-MNIST
                        |
                        |
                        v
                  DISCRIMINATOR
                        ^
                        |
                        |
                    FAKE IMAGE
                  FROM GENERATOR
            ```

            #### Step 2: Feature Extraction

            After receiving an image, the Discriminator processes it
            through neural network layers.

            It learns to identify different visual characteristics.

            **Examples of learned features:**

            - Low-level features: Edges, lines, and pixel patterns.
            - Mid-level features: Shapes, curves, and clothing silhouettes.
            - Higher-level features: More complex arrangements that help
              distinguish real training images from generated images.

            The Discriminator uses these learned features to make
            its authenticity prediction.

            #### Step 3: Classification

            After processing the image, the Discriminator produces
            a numerical value called a **Logit**.

            This logit represents the network's real-versus-fake score.

            A positive logit indicates that the network favors the real class.

            A negative logit indicates that the network favors the fake class.

            **Example:**

            | Input | Logit | Interpretation |
            |---|---:|---|
            | Real Fashion Image | +3.5 | Favors Real |
            | Generated Image | -2.8 | Favors Fake |
            | Uncertain Image | +0.1 | Slightly Favors Real |

            Important: A logit is not itself a probability.
            Applying the sigmoid function converts it into a value
            between 0 and 1.

            ---

            ### 3. Complete Discriminator Architecture

            Our project implements two Discriminator architectures:
            Basic GAN and DCGAN.

            #### A. Basic GAN Discriminator

            The Basic GAN Discriminator uses fully connected Linear layers.

            Since Fashion-MNIST images have dimensions of 28 × 28 pixels,
            each image contains 784 pixel values.

            The image is flattened before being passed to the network.

            **Architecture:**

            ```text
                INPUT IMAGE
                  1 × 28 × 28
                        |
                        v
                  FLATTEN
                  784 Values
                        |
                        v
                  LINEAR LAYER
                  784 → 1024
                        |
                        v
                    LEAKY ReLU
                        |
                        v
                  LINEAR LAYER
                  1024 → 512
                        |
                        v
                    LEAKY ReLU
                        |
                        v
                  LINEAR LAYER
                    512 → 256
                        |
                        v
                    LEAKY ReLU
                        |
                        v
                  LINEAR LAYER
                    256 → 1
                        |
                        v
                      LOGIT
                        |
                        v
                  REAL / FAKE
            ```

            **What happens at each stage?**

            1. Flatten: Converts the 28 × 28 image into 784 values.
            2. Linear layers: Learn relationships between input pixels.
            3. Leaky ReLU: Introduces non-linearity while allowing
              a small gradient for negative inputs.
            4. Final layer: Produces one real-versus-fake logit.

            #### B. DCGAN Discriminator

            DCGAN uses convolutional layers instead of relying entirely
            on fully connected layers.

            Convolutional layers examine local image regions and learn
            spatial patterns.

            **Conceptual architecture:**

            ```text
                  INPUT IMAGE
                  1 × 28 × 28
                        |
                        v
                  CONVOLUTION
                        |
                        v
                    LEAKY ReLU
                        |
                        v
                  CONVOLUTION
                        |
                        v
                    LEAKY ReLU
                        |
                        v
                  CONVOLUTION
                        |
                        v
                    LEAKY ReLU
                        |
                        v
                  FINAL OUTPUT
                        |
                        v
                        LOGIT
                        |
                        v
                    REAL / FAKE
            ```

            The exact number of convolutional layers and channel
            dimensions depends on the DCGAN implementation.

            **Why use Convolutional Layers?**

            - They learn local visual patterns.
            - They preserve spatial relationships.
            - They detect edges and shapes.
            - They are suitable for image-processing tasks.

            ---

            ### 4. What is Leaky ReLU?

            The Discriminator commonly uses an activation function
            called Leaky ReLU.

            Its mathematical formula is:

            f(x) = x, if x > 0

            f(x) = αx, if x ≤ 0

            Here, α is a small positive value, commonly 0.2.

            **Example:**

            | Input | Leaky ReLU Output (α = 0.2) |
            |---|---:|
            | -5 | -1 |
            | -2 | -0.4 |
            | 0 | 0 |
            | 3 | 3 |
            | 7 | 7 |

            **Why do we use Leaky ReLU?**

            Normal ReLU produces zero for all negative inputs.

            Leaky ReLU allows a small gradient for negative inputs,
            helping reduce the problem of neurons becoming inactive.

            ---

            ### 5. What is a Logit?

            A logit is the raw numerical output of the Discriminator
            before applying the sigmoid function.

            Our project uses logits with BCEWithLogitsLoss.

            To understand logits, consider these examples:

            | Logit | Sigmoid Probability | Interpretation |
            |---:|---:|---|
            | +3 | 0.953 | Strongly Real |
            | +1 | 0.731 | Favors Real |
            | 0 | 0.500 | Uncertain |
            | -1 | 0.269 | Favors Fake |
            | -3 | 0.047 | Strongly Fake |

            The sigmoid formula is:

            σ(x) = 1 / (1 + e^(-x))

            It converts an unrestricted logit into a value between 0 and 1.

            **Important:** Our training implementation does not need
            a separate sigmoid layer because BCEWithLogitsLoss combines
            sigmoid and binary cross-entropy in a numerically stable way.

            ---

            ### 6. Discriminator Loss Function

            The Discriminator needs to learn two things:

            1. Real images should receive a real target label (1).
            2. Fake images should receive a fake target label (0).

            Our project uses:

            **BCEWithLogitsLoss**

            BCE stands for Binary Cross Entropy.

            It measures how different the Discriminator's predictions
            are from the expected binary labels.

            The Discriminator loss is:

            L(D) = BCE(D(real), 1) + BCE(D(fake), 0)

            Where:

            - D = Discriminator
            - D(real) = Discriminator output for real images.
            - D(fake) = Discriminator output for generated images.
            - 1 = Real target.
            - 0 = Fake target.

            **Example:**

            Suppose:

            Real Image Prediction = 0.90

            Fake Image Prediction = 0.10

            Both predictions are close to their correct targets.

            Therefore, the Discriminator loss will be relatively low.

            If the Discriminator incorrectly identifies fake images
            as real, its loss increases.

            ---

            ### 7. How Does the Discriminator Learn?

            The Discriminator improves through backpropagation
            and optimization.

            **Training process:**

            1. Receive a batch of real Fashion-MNIST images.
            2. Receive fake images from the Generator.
            3. Assign real labels (1) to real images.
            4. Assign fake labels (0) to generated images.
            5. Calculate predictions using the Discriminator.
            6. Calculate Discriminator loss.
            7. Perform backpropagation.
            8. Update Discriminator weights using the optimizer.

            ```text
                    REAL IMAGES
                          |
                          v
                      D(real)
                          |
                          v
                    TARGET = 1
                          |
                          |
                          v
                      LOSS CALCULATION
                          ^
                          |
                          |
                    TARGET = 0
                          ^
                          |
                          v
                      D(fake)
                          ^
                          |
                    FAKE IMAGES
                          ^
                          |
                      GENERATOR
                          ^
                          |
                      RANDOM NOISE
                          |
                          v
                    BACKPROPAGATION
                          |
                          v
                    UPDATE D WEIGHTS
            ```

            **Important:** During the Discriminator update, fake images
            are detached from the Generator's computation graph.

            This ensures that only the Discriminator's weights are
            updated during this training step.

            ---

            ### 8. Relationship Between Generator and Discriminator

            The Generator and Discriminator have opposite objectives.

            **Generator:**
            - Creates fake fashion images.
            - Tries to fool the Discriminator.
            - Wants fake images to be classified as real.

            **Discriminator:**
            - Receives real and generated images.
            - Tries to identify fake images.
            - Wants to classify real and fake images correctly.

            This competition is called Adversarial Training.

            ```text
                      GENERATOR
                          |
                          v
                    FAKE IMAGE
                          |
                          v
                    DISCRIMINATOR
                          |
                          v
                    PREDICTION
                          |
                          v
                      LOSS
                          |
                +--------+--------+
                |                 |
                v                 v
            UPDATE G           UPDATE D
                |                 |
                v                 v
            GENERATOR          DISCRIMINATOR
            IMPROVES             IMPROVES
                |                 |
                +--------+--------+
                          |
                          v
                    REPEAT TRAINING
            ```

            Both networks gradually improve through this competition.

            ---

            ### 9. What Happens During Application Usage?

            The Discriminator is primarily used during the training phase.

            Once training is complete, we save the trained Generator
            separately.

            During normal image generation:

            - The application loads the trained Generator.
            - Random noise is provided to the Generator.
            - The Generator creates a new fashion image.
            - The generated image is displayed in Streamlit.

            The Discriminator is not required for ordinary inference.

            ```text
                      TRAINING PHASE
                            |
                    +-------+-------+
                    |               |
                    v               v
                GENERATOR     DISCRIMINATOR
                    |               |
                    +-------+-------+
                            |
                            v
                    TRAINING COMPLETE
                            |
                            v
                      SAVE GENERATOR
                            |
                            v
                      INFERENCE PHASE
                            |
                            v
                      LOAD GENERATOR
                            |
                            v
                      GENERATE IMAGE
            ```

            ---

            ### 10. Important Discriminator Parameters

            | Parameter | Basic GAN | Purpose |
            |---|---|---|
            | Input Size | 784 | Flattened image pixels |
            | Hidden Layers | 1024, 512, 256 | Learn image features |
            | Output Size | 1 | Single real/fake logit |
            | Activation | Leaky ReLU | Introduces non-linearity |
            | Loss Function | BCEWithLogitsLoss | Binary classification loss |
            | Optimizer | Adam | Updates weights |
            | Image Size | 28 × 28 | Fashion-MNIST dimensions |
            | Channels | 1 | Grayscale images |

            ---

            ### 11. Important Concepts – Quick Revision

            | Term | Explanation |
            |---|---|
            | Discriminator | Identifies real and fake images |
            | Binary Classifier | Classifies input into two categories |
            | Feature Extraction | Learns useful visual patterns |
            | Flatten | Converts image into a one-dimensional vector |
            | Convolution | Extracts spatial features |
            | Leaky ReLU | Activation function |
            | Logit | Raw output score |
            | Sigmoid | Converts logit into probability |
            | BCE | Binary Cross Entropy |
            | BCEWithLogitsLoss | Combines sigmoid and BCE |
            | Backpropagation | Calculates gradients |
            | Optimizer | Updates network weights |

            **Final Takeaway:**

            The Discriminator is the judging component of GAN FashionVerse.
            It receives real and generated fashion images, extracts their
            visual features, and produces a real-versus-fake score.

            Its feedback helps the Generator improve its ability to
            generate realistic synthetic fashion images.
            """
        )

        st.success(
            "Key takeaway: Real Image + Fake Image → Discriminator "
            "→ Real/Fake Logit → Loss → Update Discriminator."
        )

        st.warning(
            "The Discriminator is not a 10-class fashion classifier. "
            "It only learns to distinguish real training images from "
            "synthetic images generated by the Generator."
        )

        with st.expander("Explore how the Discriminator interprets logits"):
            import math

            logit = st.slider(
                "Adjust the Discriminator logit",
                min_value=-5.0,
                max_value=5.0,
                value=0.0,
                step=0.5
            )

            probability = 1 / (1 + math.exp(-logit))

            st.metric(
                "Real Probability",
                f"{probability * 100:.2f}%"
            )

            if probability > 0.5:
                st.success("The Discriminator currently favors REAL.")
            elif probability < 0.5:
                st.error("The Discriminator currently favors FAKE.")
            else:
                st.info("The Discriminator is currently uncertain.")

            st.caption(
                "This is an illustrative sigmoid conversion of a logit. "
                "It does not run your trained Discriminator model."
            )
    else:
        st.subheader("The Training Loop: How GAN Learns Through Competition")

        st.markdown(
            """
            ### 1. What is the GAN Training Loop?

            The GAN Training Loop is the process through which the
            Generator and Discriminator learn to improve their performance.

            Unlike traditional neural networks, GAN contains two networks
            that compete against each other.

            - **Generator (G):** Learns to create realistic synthetic images.
            - **Discriminator (D):** Learns to distinguish real images
              from generated images.

            Both networks are trained alternately using backpropagation
            and optimization.

            The Generator tries to fool the Discriminator, while the
            Discriminator tries to identify fake images correctly.

            This process is called **Adversarial Training**.

            ---

            ### 2. Complete GAN Training Architecture

            The following flowchart explains the complete training process.

            ```text
                        START TRAINING
                              |
                              v
                        LOAD DATASET
                              |
                              v
                      LOAD REAL IMAGE BATCH
                              |
                              v
                        SAMPLE NOISE
                              |
                              v
                          GENERATOR
                              |
                              v
                        GENERATE FAKE
                              |
                              v
                    +-------------------+
                    |                   |
                    v                   v
                REAL IMAGE          FAKE IMAGE
                    |                   |
                    +---------+---------+
                              |
                              v
                        DISCRIMINATOR
                              |
                              v
                        CALCULATE
                      DISCRIMINATOR LOSS
                              |
                              v
                        BACKPROPAGATION
                              |
                              v
                      UPDATE D WEIGHTS
                              |
                              v
                        SAMPLE NEW NOISE
                              |
                              v
                          GENERATOR
                              |
                              v
                        GENERATE FAKE
                              |
                              v
                        DISCRIMINATOR
                              |
                              v
                        CALCULATE
                        GENERATOR LOSS
                              |
                              v
                        BACKPROPAGATION
                              |
                              v
                      UPDATE G WEIGHTS
                              |
                              v
                        RECORD METRICS
                              |
                              v
                      SAVE CHECKPOINTS
                              |
                              v
                        NEXT BATCH
                              |
                              v
                        NEXT EPOCH
            ```

            ---

            ### 3. Step-by-Step GAN Training Process

            #### STEP 1: Load the Dataset

            The training process begins by loading real images from
            the Fashion-MNIST dataset.

            Our dataset contains:
            - 60,000 training images.
            - 10,000 testing images.
            - 10 different fashion categories.

            Each image has dimensions:

            1 × 28 × 28

            Images are normalized to the range [-1, 1].

            The DataLoader divides the dataset into smaller groups
            called batches.

            **What is Batch Size?**

            Batch size is the number of training images processed
            together in one training iteration.

            Example:

            If batch size = 128, the model processes 128 real images
            together.

            #### STEP 2: Generate Random Noise

            The Generator receives a random noise vector.

            In our implementation:

            Latent Dimension = 100

            Example:

            ```text
            Random Noise:

            [0.45, -0.21, 0.89, 0.12, ...]
            ```

            The Generator converts this random vector into a fake
            fashion image.

            ```text
                  RANDOM NOISE
                      |
                      v
                  GENERATOR
                      |
                      v
                FAKE IMAGE
                  1 × 28 × 28
            ```

            #### STEP 3: Train the Discriminator

            The first optimization task is to train the Discriminator.

            It receives two types of images:

            **A. Real Images**

            Images taken directly from Fashion-MNIST.

            Their target label is:

            Real = 1

            **B. Fake Images**

            Images generated by the Generator.

            Their target label is:

            Fake = 0

            The Discriminator processes both images and produces logits.

            It calculates how accurately it identifies real and fake images.

            **Discriminator Training Flow:**

            ```text
                      REAL IMAGE
                          |
                          v
                      D(real)
                          |
                          v
                      TARGET = 1
                          |
                          |
                          v
                    LOSS CALCULATION
                          ^
                          |
                          |
                      TARGET = 0
                          ^
                          |
                          v
                      D(fake)
                          ^
                          |
                      FAKE IMAGE
                          ^
                          |
                      GENERATOR
            ```

            **What is Detach()?**

            During Discriminator training, generated images are detached
            from the Generator's computation graph.

            Example:

            ```python
            fake_images = generator(noise)

            fake_images = fake_images.detach()
            ```

            Detach prevents gradients from flowing backward into the
            Generator during this optimization step.

            This ensures that only the Discriminator's parameters
            are updated.

            **Why is this necessary?**

            We want to train the Discriminator independently before
            updating the Generator.

            #### STEP 4: Calculate Discriminator Loss

            The Discriminator uses Binary Cross Entropy with Logits Loss.

            Mathematical formula:

            L(D) = BCE(D(real), 1) + BCE(D(fake), 0)

            Where:

            - D(real) = Discriminator output for real images.
            - D(fake) = Discriminator output for generated images.
            - 1 = Real target.
            - 0 = Fake target.

            The objective is to minimize the Discriminator loss.

            **Example:**

            | Input | Prediction | Correct Label |
            |---|---:|---:|
            | Real Image | 0.95 | 1 |
            | Fake Image | 0.08 | 0 |

            The predictions are close to their correct targets,
            so the Discriminator loss is relatively low.

            #### STEP 5: Backpropagation and Update Discriminator

            After calculating the loss, backpropagation calculates
            gradients for the Discriminator's parameters.

            These gradients indicate how its weights should change
            to reduce the loss.

            The optimizer then updates the weights.

            Example:

            ```python
            optimizer_D.zero_grad()

            loss_D.backward()

            optimizer_D.step()
            ```

            **Explanation:**

            - zero_grad(): Clears previously accumulated gradients.
            - backward(): Calculates gradients using backpropagation.
            - step(): Updates the Discriminator's weights.

            After this step, the Discriminator becomes better at
            distinguishing real images from generated images.

            ---

            ### 4. Training the Generator

            After updating the Discriminator, we train the Generator.

            This is the second optimization task.

            #### STEP 6: Generate a Fresh Batch of Fake Images

            The Generator receives a new random noise batch.

            ```text
                  NEW RANDOM NOISE
                          |
                          v
                      GENERATOR
                          |
                          v
                    NEW FAKE IMAGE
            ```

            This time, we do not detach the generated images.

            **Why?**

            Because we want gradients to flow backward through
            the Discriminator and reach the Generator.

            #### STEP 7: Fool the Discriminator

            The generated images are passed to the Discriminator.

            But now we use an interesting technique.

            We assign the target label:

            Real = 1

            Why?

            Because the Generator wants the Discriminator to believe
            that its fake images are real.

            Example:

            ```text
                  RANDOM NOISE
                      |
                      v
                  GENERATOR
                      |
                      v
                  FAKE IMAGE
                      |
                      v
                DISCRIMINATOR
                      |
                      v
                GENERATOR LOSS
                  TARGET = 1
            ```

            The Generator tries to maximize the Discriminator's
            confidence that the generated image is real.

            #### STEP 8: Calculate Generator Loss

            The Generator uses the non-saturating GAN loss:

            L(G) = BCE(D(G(z)), 1)

            Where:

            - G = Generator.
            - D = Discriminator.
            - z = Random noise.
            - D(G(z)) = Discriminator's output for a generated image.
            - 1 = Target label indicating real.

            The Generator wants to minimize this loss.

            A lower Generator loss generally means the Discriminator
            is assigning higher real scores to generated images.

            However, loss alone does not guarantee visual quality.

            #### STEP 9: Backpropagation and Update Generator

            Now we update the Generator's weights.

            Example:

            ```python
            optimizer_G.zero_grad()

            loss_G.backward()

            optimizer_G.step()
            ```

            **Explanation:**

            - zero_grad(): Clears the Generator's previous gradients.
            - backward(): Calculates gradients through the Discriminator
              and Generator.
            - step(): Updates only the Generator's weights.

            The Discriminator provides feedback, but its parameters
            are not updated during this Generator optimization step.

            This allows the Generator to improve its ability to
            produce realistic images.

            ---

            ### 5. Why Do We Train Both Networks Separately?

            This is one of the most important concepts in GAN.

            Both networks have different objectives.

            | Generator | Discriminator |
            |---|---|
            | Creates fake images | Examines images |
            | Wants to fool D | Wants to identify fake images |
            | Target label is 1 during G update | Real target is 1 |
            | Tries to minimize G loss | Tries to minimize D loss |
            | Learns to generate | Learns to distinguish |

            **Why alternating updates?**

            If both networks were updated incorrectly at the same time,
            their optimization objectives could interfere with each other.

            Therefore, we perform separate optimization steps.

            During the Discriminator update, only D is trained.

            During the Generator update, only G is trained.

            ---

            ### 6. What is an Epoch?

            An epoch is one complete pass through the entire
            training dataset.

            Example:

            Suppose:

            Dataset Size = 60,000 images

            Batch Size = 128

            Number of batches per epoch:

            60,000 / 128 ≈ 469 batches

            Therefore, approximately 469 training iterations
            are required to complete one epoch.

            Every iteration performs:
            1. Discriminator update.
            2. Generator update.

            If we train for 50 epochs:

            Approximately 469 × 50 = 23,450 iterations.

            Each iteration updates both networks.

            **Epoch vs Batch vs Iteration**

            | Term | Meaning |
            |---|---|
            | Batch | A group of training images |
            | Iteration | One training step using one batch |
            | Epoch | One complete pass through the dataset |

            ---

            ### 7. What Happens During Training Over Time?

            Initially, the Generator does not know how to create
            meaningful fashion images.

            Both networks start with randomly initialized weights.

            **Early Training:**

            - Generator produces random-looking images.
            - Discriminator easily detects fake images.
            - Generator receives feedback and starts learning.

            **Middle Training:**

            - Generator starts producing recognizable shapes.
            - Discriminator becomes better at identifying fake images.
            - Both networks continue improving.

            **Later Training:**

            - Generator produces more realistic clothing structures.
            - Discriminator finds it increasingly difficult to distinguish
              generated images from real images.

            ```text
                    INITIAL TRAINING
                          |
                          v
                    RANDOM OUTPUTS
                          |
                          v
                    BASIC SHAPES
                          |
                          v
                  RECOGNIZABLE CLOTHING
                          |
                          v
                  IMPROVED IMAGE QUALITY
                          |
                          v
                    TRAINED GENERATOR
            ```

            These stages are conceptual. GAN training is not always
            smooth or continuously improving.

            ---

            ### 8. What is GAN Convergence?

            Convergence refers to a state in which the Generator and
            Discriminator reach a relatively balanced adversarial state.

            Ideally:

            - The Generator produces images similar to real data.
            - The Discriminator struggles to distinguish real from fake.
            - Neither network has a significant advantage.

            In the idealized original GAN formulation, the Discriminator
            outputs approximately 0.5 for both real and generated samples.

            However, practical GANs do not always reach this ideal state.

            **Important:** A Discriminator output of 0.5 does not
            automatically mean that the Generator has learned to
            generate high-quality and diverse images.

            Visual inspection and additional evaluation metrics are useful.

            ---

            ### 9. Training Metrics and Loss Graphs

            During training, we record two important metrics.

            **A. Generator Loss**

            Measures how difficult it is for the Generator to fool
            the Discriminator.

            **B. Discriminator Loss**

            Measures how well the Discriminator distinguishes
            real and fake images.

            Both losses are displayed in the Training Dashboard.

            Example:

            | Epoch | Generator Loss | Discriminator Loss |
            |---|---:|---:|
            | 1 | 1.82 | 0.65 |
            | 5 | 1.45 | 0.82 |
            | 10 | 1.20 | 0.94 |
            | 20 | 0.95 | 1.05 |
            | 30 | 0.88 | 1.12 |

            These are illustrative values, not actual training results.

            **Important:**

            GAN losses do not necessarily decrease smoothly.

            A low loss does not always mean better images.

            Therefore, we also inspect generated sample grids
            at different epochs.

            ---

            ### 10. What is Checkpoint Saving?

            Training a GAN can take considerable time.

            Therefore, we save model checkpoints during training.

            A checkpoint contains learned model parameters.

            Example:

            ```text
            checkpoints/
                |
                |--- basic_gan_generator.pth
                |
                |--- basic_gan_discriminator.pth
                |
                |--- dcgan_generator.pth
                |
                |--- dcgan_discriminator.pth
            ```

            **Why save checkpoints?**

            - Avoid losing training progress.
            - Resume training later.
            - Compare Basic GAN and DCGAN.
            - Generate images without retraining.
            - Keep different model architectures separate.

            During inference, only the trained Generator is required
            for normal image generation.

            ---

            ### 11. Complete Training Loop in Python

            The following is a simplified example of how the
            training process works in PyTorch.

            ```python
            for epoch in range(num_epochs):

                for real_images, _ in dataloader:

                    batch_size = real_images.size(0)

                    # -------------------------
                    # 1. Train Discriminator
                    # -------------------------

                    optimizer_D.zero_grad()

                    real_labels = torch.ones(
                        batch_size, 1, device=device
                    )

                    fake_labels = torch.zeros(
                        batch_size, 1, device=device
                    )

                    noise = torch.randn(
                        batch_size, 100, device=device
                    )

                    fake_images = generator(noise)

                    real_output = discriminator(real_images)

                    fake_output = discriminator(
                        fake_images.detach()
                    )

                    loss_real = criterion(
                        real_output, real_labels
                    )

                    loss_fake = criterion(
                        fake_output, fake_labels
                    )

                    loss_D = loss_real + loss_fake

                    loss_D.backward()

                    optimizer_D.step()

                    # -------------------------
                    # 2. Train Generator
                    # -------------------------

                    optimizer_G.zero_grad()

                    noise = torch.randn(
                        batch_size, 100, device=device
                    )

                    fake_images = generator(noise)

                    fake_output = discriminator(fake_images)

                    loss_G = criterion(
                        fake_output, real_labels
                    )

                    loss_G.backward()

                    optimizer_G.step()

                print(
                    f"Epoch [{epoch+1}/{num_epochs}] "
                    f"Loss D: {loss_D.item():.4f} "
                    f"Loss G: {loss_G.item():.4f}"
                )
            ```

            This is a simplified training example. It assumes the
            models, optimizers, loss function, DataLoader, and device
            have already been initialized.

            In the actual project, model-specific handling,
            checkpoint saving, metric recording, and device management
            may be implemented separately.

            ---

            ### 12. Common Problems During GAN Training

            GAN training can be challenging.

            Some common problems include:

            **1. Mode Collapse**

            The Generator produces only a limited variety of images
            instead of generating diverse outputs.

            Example: It repeatedly generates similar-looking shoes.

            **2. Vanishing Gradients**

            The Generator receives very weak learning signals,
            making it difficult to improve.

            **3. Training Instability**

            The Generator and Discriminator may become unbalanced.

            **4. Overpowering Discriminator**

            The Discriminator becomes too good at identifying fake
            images, making it difficult for the Generator to learn.

            **5. Poor Image Quality**

            Generated images may contain distorted shapes or
            meaningless patterns.

            **How can we improve training?**

            - Adjust learning rates.
            - Experiment with batch sizes.
            - Balance Generator and Discriminator capacity.
            - Use Batch Normalization where appropriate.
            - Monitor generated sample grids.
            - Experiment with different GAN architectures.

            ---

            ### 13. Basic GAN vs DCGAN Training

            Both architectures follow the same adversarial training process.

            | Feature | Basic GAN | DCGAN |
            |---|---|---|
            | Generator | Linear Layers | Transposed Convolutions |
            | Discriminator | Linear Layers | Convolutional Layers |
            | Training Strategy | Alternating Updates | Alternating Updates |
            | Loss | BCEWithLogitsLoss | BCEWithLogitsLoss |
            | Optimizer | Adam | Adam |
            | Input Images | Fashion-MNIST | Fashion-MNIST |

            The primary difference is their network architecture,
            not their fundamental adversarial training principle.

            ---

            ### 14. Final Training Loop Summary

            Remember these steps:

            1. Load real images from the dataset.
            2. Generate fake images using random noise.
            3. Train the Discriminator using real and fake images.
            4. Calculate Discriminator loss.
            5. Update Discriminator weights.
            6. Generate a fresh batch of fake images.
            7. Train the Generator to fool the Discriminator.
            8. Calculate Generator loss.
            9. Update Generator weights.
            10. Record metrics and save checkpoints.
            11. Repeat until the training process is complete.

            **Final Takeaway:**

            GAN learns through competition. The Discriminator becomes
            better at identifying fake images, while the Generator
            becomes better at creating realistic synthetic images.

            This alternating optimization process is the heart of
            GAN FashionVerse.
            """
        )

        st.success(
            "Key takeaway: Train Discriminator → Train Generator "
            "→ Record Metrics → Repeat."
        )

        st.info(
            "One training iteration updates both networks separately. "
            "The Discriminator learns to detect fake images, while "
            "the Generator learns to fool it."
        )

        # Interactive Training Loop Demonstration

        with st.expander("Explore an Interactive GAN Training Simulation"):

            st.write(
                "Explore how the Generator and Discriminator "
                "change their objectives during alternating training."
            )

            step = st.select_slider(
                "Select a training stage",
                options=[
                    "Start",
                    "Generate Fake Images",
                    "Train Discriminator",
                    "Calculate D Loss",
                    "Update Discriminator",
                    "Train Generator",
                    "Calculate G Loss",
                    "Update Generator",
                    "Next Iteration"
                ],
                value="Start"
            )

            stages = {
                "Start": (
                    "Training initialized",
                    "Both networks are ready for training."
                ),
                "Generate Fake Images": (
                    "Generator creates fake images",
                    "Random noise is passed through the Generator."
                ),
                "Train Discriminator": (
                    "Discriminator training",
                    "Real images receive label 1 and generated images receive label 0."
                ),
                "Calculate D Loss": (
                    "Discriminator loss calculation",
                    "BCEWithLogitsLoss measures real/fake classification errors."
                ),
                "Update Discriminator": (
                    "Discriminator weights updated",
                    "Backpropagation updates only Discriminator parameters."
                ),
                "Train Generator": (
                    "Generator training",
                    "A fresh fake batch is created without detaching it."
                ),
                "Calculate G Loss": (
                    "Generator loss calculation",
                    "The Generator uses real target labels to fool the Discriminator."
                ),
                "Update Generator": (
                    "Generator weights updated",
                    "Gradients pass through D, but only Generator parameters are updated."
                ),
                "Next Iteration": (
                    "Start the next iteration",
                    "The process repeats with another batch of training images."
                )
            }

            title, description = stages[step]

            st.subheader(title)
            st.write(description)

            progress = (
                list(stages.keys()).index(step) + 1
            ) / len(stages)

            st.progress(progress)

            if step in [
                "Train Discriminator",
                "Calculate D Loss",
                "Update Discriminator"
            ]:
                st.warning(
                    "Current focus: Discriminator. "
                    "Generator weights remain unchanged."
                )

            elif step in [
                "Train Generator",
                "Calculate G Loss",
                "Update Generator"
            ]:
                st.success(
                    "Current focus: Generator. "
                    "Discriminator provides feedback but its weights remain unchanged."
                )

            else:
                st.info(
                    "This stage prepares or completes an alternating "
                    "training iteration."
                )

        with st.expander("Understand the difference between Generator and Discriminator loss"):

            st.write("Generator objective:")
            st.latex(r"L_G = BCE(D(G(z)), 1)")

            st.write("Discriminator objective:")
            st.latex(
                r"L_D = BCE(D(x), 1) + BCE(D(G(z)), 0)"
            )

            st.caption(
                "The Generator tries to make fake images look real, "
                "while the Discriminator tries to classify real and "
                "fake images correctly."
            )

        st.caption(
            "Training stability depends on learning rates, batch size, "
            "network capacity, dataset diversity, and the balance "
            "between Generator and Discriminator."
        )
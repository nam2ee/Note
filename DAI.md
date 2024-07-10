# Content
1. Decentralized AI(Utopia Architecture)
2. ZKML implementation
3. Federated Learning
4. Problem Anatomy
5. Future direction

## Decentralized AI(Utopia Architecture)
First, we must discuss what decentralized AI is, how it works, and why it matters. Currently, there are numerous D-AI projects in the crypto space.

![AIXCrypto](images/ecosystem.png)

(Thanks to AImedaResearch!)
While some of these projects are substantial, others seem quite meaningless. I believe “decentralized AI” should be synonymous with democratic AI. It shouldn’t be a network where models compete based on the quality of their inferences. Instead, users should be able to choose the best models themselves. It doesn’t necessarily need to be a blockchain system.
Nowadays, AI companies operate like dictatorships. They extract users’ private information to train their models and then sell the inference outputs. Most D-AI projects focus on proof of inference. However, I believe the focus should now shift to “training.” The training process is computationally heavier and involves significant privacy and compensation issues.
I have tolerated this so far, but I can no longer accept the loss of my data. There are articles titled “Running out of training data” ([link](https://www.makeuseof.com/ai-running-out-training-data-solutions/)). However, some are planning to extract data online in real-time, which is alarming. Therefore, AI companies will increasingly strive to collect personal information, making it a crucial issue in the future AI landscape.
To address this, decentralized AI systems based on blockchain infrastructure need to focus on one or both of the following:
- Ensuring the privacy of training data
- Providing fair compensation for training data

Discussions about open-source AI models or model ownership seem to be a next step. Personally, I see potential risks in indiscriminately releasing open-source models, such as AI jailbreaks and other forms of misuse. While I believe in the necessity of open-sourcing AI and establishing governance around it, let’s leave that for another discussion.
For now, methods to ensure the privacy of training data (known as Privacy Preserving Machine Learning) might include Federated Learning and Homomorphic Encryption. Fair compensation for training data could be achieved through the ZKML and OPML concepts in my envisioned Utopia Architecture. Here is my proposed Utopia Architecture:

![Utopia](images/utopia.png)

To be precise, this is neither an open-source model nor a privacy-preserving architecture. However, it is a practical structure that allows for compensation in exchange for sacrificing personal data, without compromising performance.
The components are as follows:
- Data Provider
- Onchain Contract(Verifier)
- Offchain model owner(Prover)
- AI Model

First, the Data Provider supplies data to the model owner (Prover) off-chain. The model owner then performs training computations on the AI model using this data and generates proof that the model was trained with the provided data and the resulting performance. Once the proof is generated, the model owner submits it to the On-chain Verifier. In return for the improvement in AI model performance, the Data Provider receives a certain amount of tokens.

## ZKML implementation
Implementing a ZKVM specifically for AI model training is crucial. Currently, there are ZK custom circuits optimized for inference. However, training is more computationally intensive and thus poses greater challenges. If training computations are successfully performed using ZK-SNARK and succinct proof of the trained model’s performance (m(t) -> m(t+1)) is submitted, rewards can be granted by the Verifier Contract on-chain. Additionally, ZK-Proof must also be generated and submitted to verify the accuracy of contribution measurements and their execution.
The On-chain Verifier submits incentives proportional to the contribution. Here is a rough code outline for the Verifier:

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^ 0.8 .0 ;

import "./Verifier.sol" ;

contract ZKMLIncentives {
Verifier public verifier;

// Data provider
address _dataProvider;

struct DataProvider {
address addr;
uint256 shapleyValue;
}

// List of data providers
DataProvider[] public dataProviders;

// Event when Shapley Value and proof is verified
event VerificationSuccess ( address indexed dataProvider, uint256 reward ) ;

constructor(address _verifier) {
verifier = Verifier(_verifier);
}

// Function to verify zk-SNARK proof and assign Shapley value
function verifyZKProof (
uint [] memory a,
uint [] memory b,
uint [] memory c,
uint256 shapleyValue
) public {
require(
verifier.verifyProof(a, b, c, input),
"Invalid zk-SNARK proof"
);

// Add data provider with verified Shapley value
DataProvider memory provider = DataProvider({
addr: msg.sender,
shapleyValue: shapleyValue
});
dataProviders.push(provider);
emit VerificationSuccess ( msg.sender, shapleyValue ) ;
}

// Function to distribute incentives based on Shapley value
function distributeIncentives () public payable {
uint256 totalShapleyValue = 0 ;
for (uint256 i = 0 ; i < dataProviders.length; i++) {
totalShapleyValue += dataProviders[i].shapleyValue;
}
require(totalShapleyValue > 0 , "No providers to reward" );

for (uint256 i = 0 ; i < dataProviders.length; i++) {
uint256 reward = (dataProviders[i].shapleyValue * msg. value ) / totalShapleyValue;
payable(dataProviders[i].addr).transfer(reward);
}
}
}
```

To introduce it properly, the formula used to measure the contribution of model training is called the Shapley Value. The Shapley Value is a method used in game theory to fairly distribute the contribution of each player in a cooperative game. The formula is expressed as follows:

![Shapley](images/shapley.png)

(Note: 𝜙(v) is the Shapley Value of provider i,
N is the set of all providers,
S is the set of providers excluding i,
v(S) is the value of the provider set S.)
This is the ideal architecture of ZKML as I envision it.

## Federated Learning
Federated Learning (FL) is a technique that allows for model training on distributed data sources while preserving data privacy. Each data provider trains a model on their local data and only sends the trained model parameters to a central server to update the global model. In this process, the data remains local, ensuring privacy.
The Process of Federated Learning is below.
Local Model Training: Each data provider trains a model on their local data. The local model update is as follows:

![Local Model](images/localmodel.png)

where w(t) are the model parameters at time t, ηis the learning rate, L is the loss function, and D is the local data of provider i .
Model Broadcast: Each data provider sends the trained local model parameters w(t+1) to the central server.
Global Model Aggregation: The central server aggregates the local model parameters received from all data providers to update the global model. The aggregation method(FedAvg, Well, another cool aggregation methods are exist) is given by (where N is the number of data providers.)

![Global Model](images/globalmodel.png)

## Problem in Federated Learning
Federated Learning faces several issues
- Performance Issues Due to Non-IID Data: When the data distributions of each data provider differ (Non-IID), the performance of the global model can degrade. This is because local models may overfit to specific data distributions. Mathematically, when the data distribution

![NonIID1](images/noniid1.png)

provider i differs, the global model’s loss function can be expressed as

![NonIID2](images/noniid2.png)

where p_i is the proportion of data from provider i . Non-IID data distributions make it challenging to optimize the global model as each

![NonIID3](images/noniid3.png)

is different.
- Communication Cost Issues: In Federated Learning, local model parameters must be sent to the central server in each round, significantly increasing communication costs. This issue becomes more severe as the number of data providers increases. The communication cost can be expressed as below. Currently, communication cost is signiture bottleneck of FedML.
- Global model stealing problem: When broadcast, global model parameters are revealed. I will protect it by encrypting global model by fixed-trainer’s public key. And then, trainers will decrypt the model using their secret key. But there are potential hazards remained.

![Communication Cost](images/communicationcost.png)

### Alternative Metric
Another method to measure contribution in Federated Learning is the ‘Gradient Norm.’ This method evaluates the contribution based on the magnitude of the gradients sent by each data provider. The formula is as follows:

![Alternative](images/alternative.png)

This method quantitatively assesses how much each data provider has contributed to model training.
Federated Learning is a powerful method for training models on distributed data sources while ensuring data privacy. However, Non-IID data and communication cost issues remain challenges. By utilizing contribution measurement methods such as Gradient Norm, a fair reward system can be established. This can enhance the efficiency of Federated Learning and encourage the participation of data providers.
Also, like ZKML, it can be used to distribute compensation rewards. But still now, there are no cool Federated learning projects.

## Problem Anatomy
Now, if we think about it carefully, it becomes clear why we face so many issues like the inability to efficiently measure contributions in AI training. Consider the Discrete Logarithm Problem (DLP) in cryptography. This problem often involves solving equations like

![DLP](images/dlp.png)

using elliptic curve groups, and the process of training AI models bears a striking resemblance to this. For instance, given

![AIDLP](images/aidlp.png)

, finding 𝑚(𝑡+1) requires vast amounts of data and significant computational resources, making it akin to a DL-Secure group. This is because it is a problem that is difficult and costly to solve.
However, issues such as data ownership and contribution measurement (with a focus on the latter here) arise primarily because ‘verification’ is challenging. How can we easily and quickly verify that
𝑚(𝑡+1) has undergone a proper training process and achieved a certain level of performance? According to the Proof of Learning paper ([link](https://arxiv.org/abs/2103.05633)), the process involves logging the training process for each batch and inspecting only a subset of parameter states. Nonetheless, in a decentralized environment like blockchain, any gaps in verification can lead to economic losses.
Thus, a more efficient and improved approach is needed. To summarize this in one sentence: we need to find points where AI training can be mapped to DLP! This is my Anatomy of the Problem. Issues like FedML’s Non-IID Problem arise inevitably in a distributed environment. Therefore, if we proceed with a clear ownership and training process, rewards must be given based on contributions. This necessitates clear ownership and Proof of Training.
“We need to find points where AI training can be mapped to DLP!”

## Future Direction
When I look at decentralized AI projects, I feel the coldness of capitalism. This is because it is too obvious that they gather various crypto narratives to pump and dump in the short term. However, from my limited perspective, the next giant that can bring a wave of change and become a hot topic in the blockchain space after DeFi is D-AI (Decentralized AI). It seems to be a field with many technical challenges, making it a good area to tackle and solve numerous issues. Therefore, I am hoping for many good projects to emerge, and I am also making efforts in this direction!
Personally, I wish that the existing VMs are too general-purpose, so I hope for a specialized layer of VM instruction sets and opcodes tailored for AI training. While I don’t expect models to be uploaded on-chain, even if they were, it would incur computational costs that even other large-scale processing L2 models cannot handle. Thus, although it may be experimental, if an AIVM (AI Virtual Machine) were to emerge, it would be quite an experimental and cool project, regardless of practicality!

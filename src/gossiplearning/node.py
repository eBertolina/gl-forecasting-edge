import math
from enum import IntEnum
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
import torch.optim as optim

from gossiplearning.config import TrainingConfig, HistoryConfig
from gossiplearning.log import Logger
from gossiplearning.models import (
    StopCriterion,
    ModelWeights,
    Loss,
    NodeId,
    MetricValue,
    MetricName,
    ModelBuilder,
    NodeDataFn,
    Dataset,
    Link,
    AggregatorFn,
    WeightsMessage,
    FisherMessage,
    LabelledData,
    NodeWeightFn,
)
from gossiplearning.weights_marshaling import MarshalWeightsFn
from utils.metrics import compute_metrics, Metrics



class NodeState(IntEnum):
    ACTIVE = 0
    STOPPED = 1
    TRAINING = 2
    FAILED = 3


class Node:
    """
    A gossip learning node.
    """

    def __init__(
        self,
        *,
        create_model_fn: ModelBuilder,
        id: NodeId,
        links: tuple[Link, ...],
        training_config: TrainingConfig,
        history_config: HistoryConfig,
        workspace_dir: Path,
        logger: Logger,
        node_data_fn: NodeDataFn,
        aggregator: AggregatorFn,
        marshal_weights_fn: MarshalWeightsFn,
        test_set: LabelledData,
        weight_fn: NodeWeightFn,
    ) -> None:
        """
        Initialize the node for gossip protocol.

        :param create_model_fn: the function used for creating a model
        :param id: the node identifier
        :param links: the set of node links
        :param training_config: the global training & gossip configuration
        :param workspace_dir: the workspace base directory
        """
        # internal state

        self._model = create_model_fn()
        self._create_model = create_model_fn
        self._logger = logger

        self._training_config = training_config
        self._history_config = history_config
        self._workspace_dir = workspace_dir

        self.data: Dataset = node_data_fn(id)

        self._last_improved_time = 0
        self._updates_without_improving = 0
        self._best_val_loss: Loss = math.inf
        self._best_weights: Optional[ModelWeights] = None
        self._completed_updates = 0

        self._received_weights: dict[NodeId, WeightsMessage] = {}
        self._aggregator = aggregator
        self._marshal_weights_fn = marshal_weights_fn
        self._test_set = test_set
        self._received_fisher = []

        # public state
        self.id = id
        self.accumulated_weight = 1
        self.active_links = list(links)
        self.training_history: dict[MetricName, list[MetricValue]] = {}
        self.state = NodeState.ACTIVE
        self.state_before_failure = NodeState.ACTIVE
        self.n_training_samples = len(self.data["X_train"])
        self.eval_metrics: list[Metrics] = []
        self.weight = weight_fn(self.data)
        self.fisher = []
        self.intermediate_fisher = []

    def merge_models(self) -> None:
        """
        Merge all the received model weights into the current model.

        The internal model weights are updated. The number of trained samples is set at the
        maximum between the number of trained samples of the merged models.
        """
        if(self._aggregator.__name__ == "merge_with_fisher_laplace_method"):
            
            if len(self.fisher) == 0:
                self.fisher = self._model.compute_fisher_diag(self.data)

            self._model, self.accumulated_weight, self.intermediate_fisher = self._aggregator(
                self._model,
                self.accumulated_weight,
                tuple(msg for k, msg in self._received_weights.items()),
                self.fisher,
                self._received_fisher
            )

        else:
            self._model, self.accumulated_weight = self._aggregator(
                self._model,
                self.accumulated_weight,
                tuple(msg for k, msg in self._received_weights.items()),
            )

        self._received_weights = {}
        self._received_fisher = []

    def perform_update(self) -> tuple[ModelWeights, ModelWeights, Loss, int]:
        """
        Perform a model update, training the node model on local data for a given number of epochs.

        The number of epochs is the one specific in the global training configuration object.

        Set the node state to TRAINING and leave it on, in order to stop reception of new models
        until the current one will be saved.

        :return: latest model weights, weights of the best trained model, its loss and the current number of updates without improvements
        """
        self.state = NodeState.TRAINING

        metrics, latest_weights, best_weights, best_val_loss = self.train_model(
            n_epochs=self._training_config.epochs_per_update,
        )

        self._completed_updates += 1
        if(self._aggregator.__name__ == "merge_with_fisher_laplace_method"):
            fisher_post_train = self._model.compute_fisher_diag(self.data)
            fisher_intermediate = np.array(self.intermediate_fisher)
            fisher_post_train = np.array(fisher_post_train)
            self.fisher = fisher_intermediate + fisher_post_train

        #self._evaluate()

        return (
            metrics,
            latest_weights,
            best_weights,
            best_val_loss,
            self._updates_without_improving,
        )

    def train_model(self, n_epochs: int) -> tuple[Metrics, ModelWeights, ModelWeights, Loss]:
        """
        Train the node model on local data for the specified number of epochs.

        After every epoch, store the training and validation metrics that will be used to build the
        training history. Also, keep track of the best obtained validation loss (among the performed
        epochs) and the related weights and return them.

        :param n_epochs: the number of training epochs.
        :return: latest model weights, weights of the best trained model and best validation loss
        """
        if n_epochs < 1:
            raise Exception("Epochs number must be at least 1!")

        model = self._create_model()
        #model = self._model
       #model.set_weights(self._model.get_weights())

        with torch.no_grad():
            model.load_state_dict(self._model.state_dict())

        best_val_loss = math.inf
        best_weights = None

        batch_size = self._training_config.batch_size
        shuffle = self._training_config.shuffle_batch
        metrics = []

        for i in range(n_epochs):

            single_epoch_metric, history = model.train_one_epoch(data = self.data,
                                            batch_size = batch_size,
                                            shuffle = shuffle, 
                                            epoch = i,
                                            id = self.id,
                                            path = self._workspace_dir,
                                            current_update = self._completed_updates,
                                            #fisher = self.fisher, 
                                            fisher = []
                                            )
            metrics.append(single_epoch_metric)
            #if len(model.loss) > 1:
            #    val_loss = sum([history[f"val_{l}_loss"][0] for l in model.loss.keys()])

            #if len(model.loss) > 1:
                #val_loss = sum([history[f"val_{l}_loss"][0] for l in model.loss.keys()])
            #else:
            val_loss = history["val_loss"][0]

            #print("OUT CYCLE Loss: ", val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = model.parameters()
                #print("Param ", i, ": ", model.parameters())

            count = 0
            #print("Test metric node.py history", history)
            for metric in history:
                if(isinstance(history[metric], list)):
                    metric_value = history[metric][0]
                else:
                    metric_value = history[metric]
                #print("Metric: ", metric)
                #print("Test metric node.py number ", count, " : ", history[metric][0])
                count = count + 1

                if metric not in self.training_history:
                    self.training_history[metric] = [metric_value]
                else:
                    self.training_history[metric].append(metric_value)

        assert best_weights
        latest_weights = model.parameters()
        return metrics, latest_weights, best_weights, best_val_loss
    
    def train_model_old(self, n_epochs: int) -> tuple[ModelWeights, ModelWeights, Loss]:
        """
        Train the node model on local data for the specified number of epochs.

        After every epoch, store the training and validation metrics that will be used to build the
        training history. Also, keep track of the best obtained validation loss (among the performed
        epochs) and the related weights and return them.

        :param n_epochs: the number of training epochs.
        :return: latest model weights, weights of the best trained model and best validation loss
        """
        if n_epochs < 1:
            raise Exception("Epochs number must be at least 1!")

        model = self._create_model()
        model.set_weights(self._model.get_weights())

        best_val_loss = math.inf
        best_weights = None
        for i in range(n_epochs):
            history = model.fit(
                self.data["X_train"],
                self.data["Y_train"],
                epochs=1,
                validation_data=(self.data["X_val"], self.data["Y_val"]),
                verbose=0,
                batch_size=self._training_config.batch_size,
                validation_batch_size=self._training_config.batch_size,
                shuffle=self._training_config.shuffle_batch,
            ).history

            if len(model.loss) > 1:
                val_loss = sum([history[f"val_{l}_loss"][0] for l in model.loss.keys()])
            else:
                val_loss = history["val_loss"][0]

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = model.parameters()

            count = 0
            #print("Test metric node.py history", history)
            for metric in history:
                metric_value = history[metric][0]
                #print("Metric: ", metric)
                #print("Test metric node.py number ", count, " : ", history[metric][0])
                count = count + 1

                if metric not in self.training_history:
                    self.training_history[metric] = [metric_value]
                else:
                    self.training_history[metric].append(metric_value)
            #print("Print loss end of iteration", self.training_history['loss'])
            #print("Print mse end of iteration", self.training_history['mape'])

        assert best_weights
        latest_weights = model.parameters()
        return latest_weights, best_weights, best_val_loss

    #def marshal_model(self) -> tuple[WeightsMessage, np.ndarray]:
    def marshal_model(self) -> WeightsMessage:
        """
        Sample weights from the current model accordingly to the percentage specified in the config.

        :return: the sampled weights
        """
        w = WeightsMessage(
            marshaled_weights=self._marshal_weights_fn(
                self._model, self._training_config.perc_sent_weights
            ),
            model_weight=self.accumulated_weight,
            optimizer_state=self._model.optimizer.variables()
            if self._training_config.serialize_optimizer
            else None,
        )
        #print(f"Marshal model print from node {self.id}: ", w.marshaled_weights.weights)
        if(self._aggregator.__name__ == "merge_with_fisher_laplace_method"):
            if len(self.fisher) == 0:
                self.fisher = self._model.compute_fisher_diag(self.data)
    
            return w, self.fisher
        else:
            return w

    def save_model(
        self,
        *,
        metrics: Metrics,          
        latest_weights: ModelWeights,
        best_update_model_weights: ModelWeights,
        time: int,
        best_update_val_loss: Loss,
        updates_without_improving: int,
        new_model_weight: int,
    ) -> None:
        """
        Update the best model and reset the early stopping counter if necessary.

        Update the best model with the received model weights, if it improved the validation loss.
        Also, update the best validation loss achieved so far in that case and reset the early
        stopping counter.

        Otherwise, increase the early stopping counter by one.
        Check if the stop criterion is met and eventually change the node state to STOPPED.

        :param best_update_model_weights: the weights of the best model trained during the last update
        :param latest_weights: the weights to be saved.
        :param time: the current time.
        :param best_update_val_loss: the validation loss achieved by the received weights
        :param updates_without_improving: the number of updates without improvements
        :param new_model_weight: the new model weight
        :return: whether the node has improved the best loss
        """
        
        #self._model.set_weights(latest_weights)

        state_dict_keys = list(self._model.state_dict().keys())

        state_dict = {
            k: w.data if isinstance(w, nn.Parameter) else w for k, w in zip(state_dict_keys, latest_weights)        
        }
        #print("state dict: ", state_dict)

        with torch.no_grad():
            self._model.load_state_dict(state_dict)
        self.accumulated_weight = new_model_weight

        improvement = self._best_val_loss - best_update_val_loss
        reset_early_stopping = improvement >= self._training_config.min_delta

        if improvement > 0:
            self._logger.debug_log(
                f"Node {self.id} improved loss by {improvement:.4f}. Early stopping is"
                f"{'' if reset_early_stopping else ' not'} reset."
            )
        else:
            self._logger.debug_log(f"Node {self.id} did not improve loss")

        self.update_best_model(best_update_model_weights, best_update_val_loss)

        # if the improvement is greater than min_delta, reset early stopping counter; otherwise,
        # increase it and eventually stop the node if the max number of epochs without improving
        # is reached
        if reset_early_stopping:
            self._updates_without_improving = 0
            self._last_improved_time = time
        else:
            self._updates_without_improving = updates_without_improving + 1
            self._logger.debug_log(
                f"This was the {self._updates_without_improving} update without improvement for node {self.id}"
            )

        satisfied_stop_criterion = self._check_stop_criterion()

        # if self.state != NodeState.FAILED:
        #     self.state = NodeState.STOPPED if satisfied_stop_criterion else NodeState.ACTIVE
        # else:
        #     self.state_before_failure = NodeState.STOPPED if satisfied_stop_criterion else NodeState.ACTIVE

        self.state = NodeState.STOPPED if satisfied_stop_criterion else NodeState.ACTIVE

    def update_best_model(self, weights: ModelWeights, val_loss: float):
        if val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            self._best_weights = weights

    def persist_best_model(self) -> None:
        """
        Serialize and persist the best model achieved so far.
        """
        best_model = self._create_model()
        #best_model.set_weights(self._best_weights)

        state_dict_keys = list(best_model.state_dict().keys())

        #state_dict = {
        #    k: torch.from_numpy(w) for k, w in zip(state_dict_keys, self._best_weights)
        #}

        state_dict = {
            k: w.data if isinstance(w, nn.Parameter) else w for k, w in zip(state_dict_keys, self._best_weights)        
        }

        with torch.no_grad():
            best_model.load_state_dict(state_dict)
        
        #best_model.save(
        #    str(
        #        self._workspace_dir
        #        / self._training_config.models_folder
        #        / f"{self.id}.h5"
        #    )
        #)

        model_folder = Path(self._workspace_dir) / self._training_config.models_folder
        model_folder.mkdir(parents=True, exist_ok=True)

        torch.save(best_model.state_dict(),
                    str(
                       model_folder
                       / f"{self.id}.pth"
                    )
        )

    def receive_weights(self, received: WeightsMessage, from_node: NodeId, 
                        fisher_mtx: np.ndarray) -> None:
        """
        Receive marshaled weights from a node and store them in the internal buffer.

        :param received: the received weights
        :param from_node: the node from which the weights came from
        """
        self._received_weights[from_node] = received
        #print(f"Received weights from node {from_node} to node {self.id} ", received.marshaled_weights.weights)
        #print(f"Received matrix from node {from_node} to node {self.id} ", fisher_mtx)
        self._received_fisher = fisher_mtx
        #print(f"Self received ", self._received_fisher)
        #print(f"Self local ", self.fisher)

    def _evaluate(self) -> None:
        #print("Check eval")
        if (
            self._history_config.eval_test
            and self._completed_updates % self._history_config.freq == 0
        ):
            X, Y = self._test_set
            pred = self._model.predict(X, verbose=0)

            metrics = compute_metrics(Y, pred)
            self.eval_metrics.append(metrics)

    @property
    def ready_to_train(self) -> bool:
        """
        Whether then node has buffered a minimum number of models to perform an update.
        """
        return len(self._received_weights) >= self._training_config.num_merged_models

    def _check_stop_criterion(self):
        """
        Check if the stop criterion is met and eventually stop the node.
        """
        if self._training_config.stop_criterion == StopCriterion.NO_IMPROVEMENTS:
            satisfied_stop_criterion = (
                self._updates_without_improving >= self._training_config.patience
            )
        elif self._training_config.stop_criterion == StopCriterion.FIXED_UPDATES:
            satisfied_stop_criterion = (
                self._completed_updates >= self._training_config.fixed_updates
            )
        else:
            raise Exception("Unrecognized stop criterion!")
        
        return satisfied_stop_criterion
    

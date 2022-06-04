## Code adapted from https://github.com/cgraywang/deepex/blob/main/src/deepex/model/kgm.py

import torch
import logging
from typing import Callable, Dict, List, Optional, Tuple


def predict_and_save_results(dataloader: DataLoader, description: str, trainer,
                             model_args, tokenizer, prediction_loss_only: Optional[bool] = None
                             ):
    if model_args.compute_loss:
        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else trainer.prediction_loss_only

    model = trainer.model
    if trainer.args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    else:
        model = trainer.model

    batch_size = dataloader.batch_size
    logger.info("***** Running %s *****", description)
    logger.info("  Num examples = %d", trainer.num_examples(dataloader))
    logger.info("  Batch size = %d", batch_size)
    if model_args.compute_loss:
        eval_losses: List[float] = []
        preds: torch.Tensor = None
        label_ids: torch.Tensor = None
    model.eval()

    if is_torch_tpu_available():
        dataloader = pl.ParallelLoader(dataloader, [trainer.args.device]).per_device_loader(trainer.args.device)

    res_dict = {}
    res_rel_dict = {}
    search_res = {}
    stats = {'max': -1, 'min': 1, 'sum': 0, 'num': 0, 'plot': None}
    for inputs in tqdm(dataloader, desc=description):
        if model_args.compute_loss:
            has_labels = any(inputs.get(k) is not None for k in ["labels", "lm_labels", "masked_lm_labels"])
        entity_ids = inputs.pop('entity_ids')
        head_entity_ids = inputs.pop('head_entity_ids')
        tail_entity_ids = inputs.pop('tail_entity_ids')
        relation_entity_ids = inputs.pop('relation_entity_ids')
        special_tokens_mask = inputs.pop('special_tokens_mask')
        docid = inputs.pop('docid')
        offset = inputs.pop('offset')
        text = inputs.pop('text')
        for k, v in inputs.items():
            inputs[k] = v.to(trainer.args.device)

        with torch.no_grad():
            tic = time.time()
            outputs = model(**inputs)
            logger.info('forward time cost {}s'.format(time.time() - tic))
            for k, v in inputs.items():
                inputs[k] = v.cpu()
            inputs['entity_ids'] = entity_ids
            inputs['head_entity_ids'] = head_entity_ids
            inputs['tail_entity_ids'] = tail_entity_ids
            inputs['relation_entity_ids'] = relation_entity_ids
            inputs['special_tokens_mask'] = special_tokens_mask
            inputs['docid'] = docid
            inputs['offset'] = offset
            inputs['text'] = text
            if model_args.generation_type == 'fast_unsupervised_bidirectional_beam_search':
                attentions = transform_layer_attention(layer_attention(outputs[-1], model_args.search_layer_id),
                                                       model_args.search_attention_head_type)
                merge_search_res(fast_unsupervised_bidirectional_beam_search(attentions, model_args, inputs, tokenizer),
                                 search_res)
            else:
                raise ValueError('search not supported')
            if model_args.compute_loss:
                if has_labels:
                    step_eval_loss, logits = outputs[:2]
                    eval_losses += [step_eval_loss.mean().item()]
                else:
                    logits = outputs[0]
        if model_args.compute_loss:
            if not prediction_loss_only:
                if preds is None:
                    preds = logits.detach()
                else:
                    preds = torch.cat((preds, logits.detach()), dim=0)
                if inputs.get("labels") is not None:
                    if label_ids is None:
                        label_ids = inputs["labels"].detach()
                    else:
                        label_ids = torch.cat((label_ids, inputs["labels"].detach()), dim=0)
    if model_args.compute_loss:
        if trainer.args.local_rank != -1:
            if preds is not None:
                preds = trainer.distributed_concat(preds, num_total_examples=trainer.num_examples(dataloader))
            if label_ids is not None:
                label_ids = trainer.distributed_concat(label_ids, num_total_examples=trainer.num_examples(dataloader))
        elif is_torch_tpu_available():
            if preds is not None:
                preds = xm.mesh_reduce("eval_preds", preds, torch.cat)
            if label_ids is not None:
                label_ids = xm.mesh_reduce("eval_label_ids", label_ids, torch.cat)

        if preds is not None:
            preds = preds.cpu().numpy()
        if label_ids is not None:
            label_ids = label_ids.cpu().numpy()

        if trainer.compute_metrics is not None and preds is not None and label_ids is not None:
            metrics = trainer.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
        else:
            metrics = {}
        if len(eval_losses) > 0:
            metrics["eval_loss"] = np.mean(eval_losses)

        for key in list(metrics.keys()):
            if not key.startswith("eval_"):
                metrics[f"eval_{key}"] = metrics.pop(key)
    res_dict = sorted(res_dict.items(), key=lambda x: x[1], reverse=True)
    if model_args.compute_loss:
        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics), \
               (res_dict, res_rel_dict, stats, search_res)
    return None, (res_dict, res_rel_dict, stats, search_res)
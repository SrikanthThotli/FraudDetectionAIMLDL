from modelValidation import modelSimulator

model = modelSimulator()

#model.train_perf_test_perf()

model.holdOutValidation()
#
model.prequentialValidation()
#
model.prequentialSplit()
#
# model.card_precision_top_k_custom()
#
# model.gridSearch()
#
# model.integration()
#
# model.modelSelection()
#
# model.modelSelLogisticReg()
#
# model.modelSelRanForClass()
#
# model.modelSelXgboost()
#

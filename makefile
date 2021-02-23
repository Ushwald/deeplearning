EPOCHS = 50

crossvalidation_experiments:
	for LR in 0.001 0.01 0.05 0.1 ; do \
		for MOM in 0 0.1 0.5 1.0 ; do \
			python main.py --epochs=$(EPOCHS) --crossvalidation=True --learningrate=$$LR --momentum=$$MOM ;\
			python main.py --epochs=$(EPOCHS) --crossvalidation=True --learningrate=$$LR --momentum=$$MOM --optimizer=sgdm ;\
			python main.py --epochs=$(EPOCHS) --crossvalidation=True --learningrate=$$LR --momentum=$$MOM --optimizer=sgdm -- activation=elu ;\
			python main.py --epochs=$(EPOCHS) --crossvalidation=True --learningrate=$$LR --momentum=$$MOM -- activation=elu ;\
		done ;\
	done

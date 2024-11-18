history:
	git log --graph --decorate --oneline --all

move:
	mv results/*.pkl results/raw/

download:
	# Download results
	rsync -avz csf:"~/Neurips/results/*" ./results/

upload:
	rsync -avz "Neurips/results/20240707_5103109/*.o5046499.*" csf:"~/Neurips/results/20240707_5103109/"

view-task-mem:
	qacct -j <job_id> -t <task_id> | grep maxvmem

# Use the make command with: make convert NOTEBOOK=my_notebook.ipynb
convert:
	jupyter nbconvert --to script $(NOTEBOOK)
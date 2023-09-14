export CUDA_VISIBLE_DEVICES=4
dir=L9_K8
for j in {95..116}
do
	#echo -e "\n\n\n\nCheckpoint${j}"
	#checkpoint="checkpoints/"${dir}"/checkpoint${j}.pt"
	#echo -e "\n\n\n\n$file"

	file=checkpoint${j}.pt
	echo -e "\n$file" >> /data/guoshoutao/HMT_glancing_futture/checkpoints/${dir}/AL.txt

	

	#python scripts/average_checkpoints.py --inputs $modelfile --num-update-checkpoints ${j} --output $modelfile/average-model.pt 

	#python scripts/average_checkpoints3.py --inputs $modelfile --num-update-checkpoints 3 --output $modelfile/average-model.pt --last_file $file
	#file=$modelfile/average-model.pt

	#python scripts/average_checkpoints.py --inputs $modelfile --num-update-checkpoints ${j} --output $modelfile/average-model.pt --last_file $file
	#python scripts/average_checkpoints.py --inputs $modelfile --num-epoch-checkpoints ${j} --checkpoint-upper-bound 48 --output $modelfile/average-model.pt

	
	for testk in 1
	do
		#echo -e "\ndecode with wait-${testk}"
		
		#python generate.py data-bin/wmt15_de_en_bpe32k --path ${checkpoint} --batch-size 250 --beam 1 --left-pad-source False --fp16 --lenpen 1.4 --remove-bpe > ./res/${dir}/${name} 2>&1
		#python fairseq_cli/generate.py data-bin/wmt15_de_en_bpe32k --path $modelfile/average-model.pt --batch-size 250 --beam 1 --lenpen 1.4 --remove-bpe  > ./res/${dir}/${name} 2>&1

		python /data/guoshoutao/HMT_glancing_futture/generate.py /data/guoshoutao/wmt15_de_en_bpe32k --path /data/guoshoutao/HMT_glancing_futture/checkpoints/${dir}/${file} --batch-size 100 --beam 1 --left-pad-source False --fp16 --remove-bpe > /data/guoshoutao/HMT_glancing_futture/checkpoints/${dir}/ress_mt 2>&1
		tail -n 2 /data/guoshoutao/HMT_glancing_futture/checkpoints/${dir}/ress_mt >> /data/guoshoutao/HMT_glancing_futture/checkpoints/${dir}/AL.txt
		
		grep ^H /data/guoshoutao/HMT_glancing_futture/checkpoints/${dir}/ress_mt | cut -f1,3- | cut -c3- | sort -k1n | cut -f2- > /data/guoshoutao/HMT_glancing_futture/checkpoints/${dir}/ress_mt.translation
		/data/guoshoutao/multi-bleu.perl -lc  /data/guoshoutao/MMA/MMA/raw_datasets/wmt15_de_en/newstest2015.tok.en < /data/guoshoutao/HMT_glancing_futture/checkpoints/${dir}/ress_mt.translation >> /data/guoshoutao/HMT_glancing_futture/checkpoints/${dir}/AL.txt
		
	done
done
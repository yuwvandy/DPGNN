echo "=====CORA====="
echo "DPGCN w/o SSL + LP"
python main.py --dataset='Cora' --shuffle='eval_total' --encoder='GCN' --label_prop='no' --ssl='no' --episodic_samp=1 --runs=20 --eta=3 --lamb1=10 --lamb2=10


echo "DPGCN w/o SSL"
python main.py --dataset='Cora' --shuffle='eval_total' --encoder='GCN' --label_prop='yes' --ssl='no' --episodic_samp=1 --runs=20 --eta=3 --lamb1=10 --lamb2=10


echo "DPGCN w/o LP"
python main.py --dataset='Cora' --shuffle='eval_total' --encoder='GCN' --label_prop='no' --ssl='yes' --episodic_samp=1 --runs=20 --eta=3 --lamb1=10 --lamb2=10


echo "DPGCN"
python main.py --dataset='Cora' --shuffle='eval_total' --encoder='GCN' --label_prop='yes' --ssl='yes' --episodic_samp=1 --runs=20 --eta=3 --lamb1=10 --lamb2=10


echo "=====CITESEER====="
echo "DPGCN w/o SSL + LP"
python main.py --dataset='Citeseer' --shuffle='eval_total' --encoder='GCN' --label_prop='no' --ssl='no' --episodic_samp=1 --runs=20 --eta=5 --lamb1=10 --lamb2=10


echo "DPGCN w/o SSL"
python main.py --dataset='Citeseer' --shuffle='eval_total' --encoder='GCN' --label_prop='yes' --ssl='no' --episodic_samp=1 --runs=20 --eta=5 --lamb1=10 --lamb2=10


echo "DPGCN w/o LP"
python main.py --dataset='Citeseer' --shuffle='eval_total' --encoder='GCN' --label_prop='no' --ssl='yes' --episodic_samp=1 --runs=20 --eta=5 --lamb1=10 --lamb2=10


echo "DPGCN"
python main.py --dataset='Citeseer' --shuffle='eval_total' --encoder='GCN' --label_prop='yes' --ssl='yes' --episodic_samp=1 --runs=20 --eta=5 --lamb1=10 --lamb2=10




echo "=====PUBMED====="
echo "DPGCN w/o SSL + LP"
python main.py --dataset='Pubmed' --shuffle='eval_total' --encoder='GCN' --label_prop='no' --ssl='no' --episodic_samp=1 --runs=20 --eta=1 --lamb1=10 --lamb2=10


echo "DPGCN w/o SSL"
python main.py --dataset='Pubmed' --shuffle='eval_total' --encoder='GCN' --label_prop='yes' --ssl='no' --episodic_samp=1 --runs=20 --eta=1 --lamb1=10 --lamb2=10


echo "DPGCN w/o LP"
python main.py --dataset='Pubmed' --shuffle='eval_total' --encoder='GCN' --label_prop='no' --ssl='yes' --episodic_samp=1 --runs=20 --eta=1 --lamb1=10 --lamb2=10


echo "DPGCN"
python main.py --dataset='Pubmed' --shuffle='eval_total' --encoder='GCN' --label_prop='yes' --ssl='yes' --episodic_samp=1 --runs=20 --eta=1 --lamb1=10 --lamb2=10



echo "=====Cora_ML====="
echo "DPGCN w/o SSL + LP"
python main.py --dataset='Cora_ML' --shuffle='eval_total' --encoder='GCN' --label_prop='no' --ssl='no' --episodic_samp=1 --runs=20 --eta=2 --lamb1=1 --lamb2=1


echo "DPGCN w/o SSL"
python main.py --dataset='Cora_ML' --shuffle='eval_total' --encoder='GCN' --label_prop='yes' --ssl='no' --episodic_samp=1 --runs=20 --eta=2 --lamb1=1 --lamb2=1


echo "DPGCN w/o LP"
python main.py --dataset='Cora_ML' --shuffle='eval_total' --encoder='GCN' --label_prop='no' --ssl='yes' --episodic_samp=1 --runs=20 --eta=2 --lamb1=1 --lamb2=1


echo "DPGCN"
python main.py --dataset='Cora_ML' --shuffle='eval_total' --encoder='GCN' --label_prop='yes' --ssl='yes' --episodic_samp=1 --runs=20 --eta=2 --lamb1=1 --lamb2=1



echo "=====DBLP====="
echo "DPGCN w/o SSL + LP"
python main.py --dataset='DBLP' --shuffle='eval_total' --encoder='GCN' --label_prop='no' --ssl='no' --episodic_samp=1 --runs=20 --eta=1 --lamb1=10 --lamb2=10


echo "DPGCN w/o SSL"
python main.py --dataset='DBLP' --shuffle='eval_total' --encoder='GCN' --label_prop='yes' --ssl='no' --episodic_samp=1 --runs=20 --eta=1 --lamb1=10 --lamb2=10


echo "DPGCN w/o LP"
python main.py --dataset='DBLP' --shuffle='eval_total' --encoder='GCN' --label_prop='no' --ssl='yes' --episodic_samp=1 --runs=20 --eta=1 --lamb1=10 --lamb2=10


echo "DPGCN"
python main.py --dataset='DBLP' --shuffle='eval_total' --encoder='GCN' --label_prop='yes' --ssl='yes' --episodic_samp=1 --runs=20 --eta=1 --lamb1=10 --lamb2=10



echo "=====Computers====="
echo "DPGCN w/o SSL + LP"
python main.py --dataset='computers' --shuffle='eval_total' --encoder='GCN' --label_prop='no' --ssl='no' --episodic_samp=1 --runs=20 --eta=3 --lamb1=1 --lamb2=1


echo "DPGCN w/o SSL"
python main.py --dataset='computers' --shuffle='eval_total' --encoder='GCN' --label_prop='yes' --ssl='no' --episodic_samp=1 --runs=20 --eta=3 --lamb1=1 --lamb2=1


echo "DPGCN w/o LP"
python main.py --dataset='computers' --shuffle='eval_total' --encoder='GCN' --label_prop='no' --ssl='yes' --episodic_samp=1 --runs=20 --eta=3 --lamb1=1 --lamb2=1


echo "DPGCN"
python main.py --dataset='computers' --shuffle='eval_total' --encoder='GCN' --label_prop='yes' --ssl='yes' --episodic_samp=1 --runs=20 --eta=3 --lamb1=1 --lamb2=1



echo "=====Photo====="
echo "DPGCN w/o SSL + LP"
python main.py --dataset='photo' --shuffle='eval_total' --encoder='GCN' --label_prop='no' --ssl='no' --episodic_samp=1 --runs=20 --eta=4 --lamb1=10 --lamb2=10


echo "DPGCN w/o SSL"
python main.py --dataset='photo' --shuffle='eval_total' --encoder='GCN' --label_prop='yes' --ssl='no' --episodic_samp=1 --runs=20 --eta=4 --lamb1=10 --lamb2=10


echo "DPGCN w/o LP"
python main.py --dataset='photo' --shuffle='eval_total' --encoder='GCN' --label_prop='no' --ssl='yes' --episodic_samp=1 --runs=20 --eta=4 --lamb1=10 --lamb2=10


echo "DPGCN"
python main.py --dataset='photo' --shuffle='eval_total' --encoder='GCN' --label_prop='yes' --ssl='yes' --episodic_samp=1 --runs=20 --eta=4 --lamb1=10 --lamb2=10



echo "=====Twitter_PT====="
echo "DPGCN w/o SSL + LP"
python main.py --dataset='PT' --shuffle='eval_total' --encoder='GCN' --label_prop='no' --ssl='no' --episodic_samp=1 --runs=20 --eta=1 --lamb1=10 --lamb2=10


echo "DPGCN w/o SSL"
python main.py --dataset='PT' --shuffle='eval_total' --encoder='GCN' --label_prop='yes' --ssl='no' --episodic_samp=1 --runs=20 --eta=1 --lamb1=10 --lamb2=10


echo "DPGCN w/o LP"
python main.py --dataset='PT' --shuffle='eval_total' --encoder='GCN' --label_prop='no' --ssl='yes' --episodic_samp=1 --runs=20 --eta=1 --lamb1=10 --lamb2=10


echo "DPGCN"
python main.py --dataset='PT' --shuffle='eval_total' --encoder='GCN' --label_prop='yes' --ssl='yes' --episodic_samp=1 --runs=20 --eta=1 --lamb1=10 --lamb2=10

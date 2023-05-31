echo "=====CORA====="
echo "DPGCN"
for ratio in 1 2 4 6 8 10
do
    echo "ratio 1:"$ratio
    python main.py --dataset='Cora' --shuffle='eval_total' --encoder='GCN' --label_prop='yes' --ssl='yes' --episodic_samp=1 --runs=20 --eta=3 --lamb1=10 --lamb2=10 --imb_ratio=$ratio
done

echo "=====CITESEER====="
echo "DPGCN"
for ratio in 1 2 4 6 8 10
do
    echo "ratio 1:"$ratio
    python main.py --dataset='Citeseer' --shuffle='eval_total' --encoder='GCN' --label_prop='yes' --ssl='yes' --episodic_samp=1 --runs=20 --eta=5 --lamb1=10 --lamb2=10 --imb_ratio=$ratio
done



echo "=====PUBMED====="
echo "DPGCN"
for ratio in 1 2 4 6 8 10
do
    echo "ratio 1:"$ratio
    python main.py --dataset='Pubmed' --shuffle='eval_total' --encoder='GCN' --label_prop='yes' --ssl='yes' --episodic_samp=1 --runs=20 --eta=1 --lamb1=10 --lamb2=10 --imb_ratio=$ratio
done


echo "=====Cora_ML====="
echo "DPGCN"
for ratio in 1 2 4 6 8 10
do
    echo "ratio 1:"$ratio
    python main.py --dataset='Cora_ML' --shuffle='eval_total' --encoder='GCN' --label_prop='yes' --ssl='yes' --episodic_samp=1 --runs=20 --eta=2 --lamb1=1 --lamb2=1 --imb_ratio=$ratio
done


echo "=====DBLP====="
echo "DPGCN"
for ratio in 1 2 4 6 8 10
do
    echo "ratio 1:"$ratio
    python main.py --dataset='DBLP' --shuffle='eval_total' --encoder='GCN' --label_prop='yes' --ssl='yes' --episodic_samp=1 --runs=20 --eta=1 --lamb1=10 --lamb2=10 --imb_ratio=$ratio
done


echo "=====Computers====="
echo "DPGCN"
for ratio in 1 2 4 6 8 10
do
    echo "ratio 1:"$ratio
    python main.py --dataset='computers' --shuffle='eval_total' --encoder='GCN' --label_prop='yes' --ssl='yes' --episodic_samp=1 --runs=20 --eta=3 --lamb1=1 --lamb2=1 --imb_ratio=$ratio
done


echo "=====Photo====="
echo "DPGCN"
for ratio in 1 2 4 6 8 10
do
    echo "ratio 1:"$ratio
    python main.py --dataset='photo' --shuffle='eval_total' --encoder='GCN' --label_prop='yes' --ssl='yes' --episodic_samp=1 --runs=20 --eta=4 --lamb1=10 --lamb2=10 --imb_ratio=$ratio
done


echo "=====Twitter_PT====="
echo "DPGCN"
for ratio in 1 2 4 6 8 10
do
    echo "ratio 1:"$ratio
    python main.py --dataset='PT' --shuffle='eval_total' --encoder='GCN' --label_prop='yes' --ssl='yes' --episodic_samp=1 --runs=20 --eta=1 --lamb1=10 --lamb2=10 --imb_ratio=$ratio
done

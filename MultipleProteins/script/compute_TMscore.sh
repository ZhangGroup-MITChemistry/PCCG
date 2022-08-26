# for name in $(ls ./md/*/*.pdb); do
#     echo $name
#     TMscore $name ref.pdb | grep "^TM-score" >> ./TM_score_md.txt
# done

for name in $(ls ./cg/*/*.pdb); do
    echo $name
    TMscore $name ref.pdb | grep "^TM-score" >> ./TM_score_cg.txt
done

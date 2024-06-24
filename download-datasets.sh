function git_sparse_clone() (
    rurl="$1" localdir="$2" subdir="$3" && shift 3

    git clone -n --depth=1 --filter=tree:0 \
        "$rurl" "$localdir"
    cd "$localdir"
    git sparse-checkout set --no-cone "$subdir"
    git checkout

    mv .$subdir/* .
    rm -r .$subdir
)

git_sparse_clone "https://github.com/IndoNLP/nusax" "data/nusax" "/datasets/mt"
rm -r data/nusax/datasets/

git_sparse_clone "https://github.com/IndoNLP/nusa-writes" "data/nusa-writes" "/data"

git_sparse_clone "https://github.com/openlanguagedata/seed" "data/seed" "/seed"
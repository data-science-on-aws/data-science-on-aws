aws s3 sync \
    . \
    s3://dsoaws/workshop/ \
    --follow-symlinks \
    --delete \
    --acl public-read-write \
    --acl bucket-owner-full-control \
    --exclude "*" \
    --include "lab.template" \
    --include "lab.policy"  
#    --include "00_quickstart/*" \

echo "DON'T FORGET TO UPDATE THE PERMISSIONS ON THE FILES!!!!"

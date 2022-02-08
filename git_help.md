### HELP GITHUB MELD_CLASSIFIER_GDL

## Clone folder

1) Git clone
```bash
git clone https://github.com/mathrip/meld_classifier_GDL.git
```

2) Go into folder
```bash
cd meld_classifier_GDL
```

3) Check your remote
```bash
git remote -v 
```
It should say :
```
origin	https://github.com/mathrip/meld_classifier_GDL.git (fetch)
origin	https://github.com/mathrip/meld_classifier_GDL.git (push)
```

4) check in which branch you are
```bash
git branch
```
It should say:
```
* main
```

## Create and work on new branch
#Before creating a new branch, pull the changes from upstream. Your master needs to be up to date.
```bash
git pull
```

#Create the branch on your local machine and switch on this branch :
```bash
git checkout -b [name_of_your_new_branch]
```

#Push the branch on github :
```bash
git push origin [name_of_your_new_branch]
```

#commit a change on the branch
```bash
git commit -m 'description change'
```

#push change on origin/branch
```bash
git push origin [name_of_your_new_branch]
```

## Work on an existing branch from a remote

#checkout on a remote branch (from origin for example)
```bash
git checkout --track origin/[branch_name]
```

#Commit, push, etc are the same than above

## Other tricks

#Switch from branch A to branch B:
(Note: you need to have push your changes from branch A before to go on branch B)
```bash
git checkout [name_of_branch_B]
```

#To see local branches, run this command: 
```bash
git branch
```

#To see remote branches, run this command:
```bash
git branch -r.
```

#To see all local and remote branches, run this command:
```bash
git branch -a.
```

#Delete a branch on your local filesystem :
```bash
git branch -d [name_of_your_new_branch]
```

#add commit from branch A to branch B
```bash
git checkout B
git cherry-pick <commit_NumberID>
```

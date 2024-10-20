install:
    pip install -r requirements.txt

format:
    black *.py

train:
    python train.py

eval:
    python -m pytest tests.py  # if tests are available

update-branch:
    git config --global user.name $(NaveedKhanBaloch)
    git config --global user.email $(naveedk09@gmail.com)
    git commit -am "Update with new results"
    git push --force origin HEAD:update

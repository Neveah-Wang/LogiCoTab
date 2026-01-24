@echo off
@REM python main_vae(new).py


@REM python main_ddpm.py --config exp/page/CoTable/config.toml
@REM python main_ddpm.py --config exp/yeast_me2/CoTable/config.toml
@REM python main_ddpm.py --config exp/winequality/CoTable/config.toml

@REM python filter.py --config exp/bean/CoTable/config.toml > evaluate/mle_log(AucF1AccGmeanMcc)/bean.log
@REM python filter.py --config exp/page/CoTable/config.toml > evaluate/mle_log(AucF1AccGmeanMcc)/page.log
@REM python filter.py --config exp/obesity/CoTable/config.toml > evaluate/mle_log(AucF1AccGmeanMcc)/obesity.log
@REM python filter.py --config exp/yeast_me2/CoTable/config.toml > evaluate/mle_log(AucF1AccGmeanMcc)/yeast_me2.log
@REM python filter.py --config D:/Study/自学/表格数据生成/LogiCoTab-vae/exp/winequality/CoTable/config.toml > evaluate/mle_log(AucF1AccGmeanMcc)/winequality.log



@REM echo magic
@REM python baselines/TabDDPM/main.py --config exp/magic/TabDDPM/config.toml --eval
@REM echo churn
@REM python baselines/TabDDPM/main.py --config exp/churn/TabDDPM/config.toml --train --sample --eval
@REM echo shopper
@REM python baselines/TabDDPM/main.py --config exp/shopper/TabDDPM/config.toml --train --sample --eval
@REM echo obesity
@REM python baselines/TabDDPM/main.py --config exp/obesity/TabDDPM/config.toml --train --sample --eval
@REM echo winequality
@REM python baselines/TabDDPM/main.py --config exp/winequality/TabDDPM/config.toml --train --sample --eval
@REM echo bean
@REM python baselines/TabDDPM/main.py --config exp/bean/TabDDPM/config.toml --eval
@REM echo yeast_me2
@REM python baselines/TabDDPM/main.py --config exp/yeast_me2/TabDDPM/config.toml --eval
@REM echo page
@REM python baselines/TabDDPM/main.py --config exp/page/TabDDPM/config.toml --train --sample --eval
@REM echo buddy
@REM python baselines/TabDDPM/main.py --config exp/buddy/TabDDPM/config.toml --train --sample --eval


@REM echo adult
@REM python baselines/TabSyn/main.py --config exp/adult/TabSyn/config.toml --train --sample --eval
@REM echo magic
@REM python baselines/TabSyn/main.py --config exp/magic/TabSyn/config.toml --train --sample --eval
@REM echo churn
@REM python baselines/TabSyn/main.py --config exp/churn/TabSyn/config.toml --train --sample --eval
@REM echo shopper
@REM python baselines/TabSyn/main.py --config exp/shopper/TabSyn/config.toml --train --sample --eval
@REM echo obesity
@REM python baselines/TabSyn/main.py --config exp/obesity/TabSyn/config.toml --train --sample --eval
@REM echo bean
@REM python baselines/TabSyn/main.py --config exp/bean/TabSyn/config.toml --train --sample --eval
echo page
python baselines/TabSyn/main.py --config exp/page/TabSyn/config.toml --train --sample --eval
echo buddy
python baselines/TabSyn/main.py --config exp/buddy/TabSyn/config.toml --train --sample --eval
@REM echo winequality
@REM python baselines/TabSyn/main.py --config exp/winequality/TabSyn/config.toml --train --sample --eval
@REM echo yeast_me2
@REM python baselines/TabSyn/main.py --config exp/yeast_me2/TabSyn/config.toml --train --sample --eval


python baselines/CTGAN_TVAE/main_ctgan.py --config exp/adult/CTGAN/config.toml --train --sample --eval
python baselines/CTGAN_TVAE/main_ctgan.py --config exp/magic/CTGAN/config.toml --train --sample --eval
python baselines/CTGAN_TVAE/main_ctgan.py --config exp/churn/CTGAN/config.toml --train --sample --eval
python baselines/CTGAN_TVAE/main_ctgan.py --config exp/shopper/CTGAN/config.toml --train --sample --eval
python baselines/CTGAN_TVAE/main_ctgan.py --config exp/obesity/CTGAN/config.toml --train --sample --eval
python baselines/CTGAN_TVAE/main_ctgan.py --config exp/bean/CTGAN/config.toml --train --sample --eval
python baselines/CTGAN_TVAE/main_ctgan.py --config exp/page/CTGAN/config.toml --train --sample --eval
python baselines/CTGAN_TVAE/main_ctgan.py --config exp/buddy/CTGAN/config.toml --train --sample --eval
python baselines/CTGAN_TVAE/main_ctgan.py --config exp/winequality/CTGAN/config.toml --train --sample --eval
python baselines/CTGAN_TVAE/main_ctgan.py --config exp/yeast_me2/CTGAN/config.toml --train --sample --eva
echo All tasks completed!
pause
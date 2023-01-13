	#!/bin/sh

	### 老板你的代码不在隐藏目录下或者本身不是隐藏文件，那就改以下几行就可以了。
	### 如果是隐藏目录下的文件或者本身是隐藏文件，则需要修改下38~53行的内容
	### 以句点"."开头的文件夹和文件称为隐藏目录和隐藏文件，如.myDir和.config.ts
	#### ------------------------------------------------------------------
	## 版本V3 通过fortify扫描typescript模板
	export PROJECT_NAME="DTSE-HDN.-VERSION.JS"
	## 跟codedex上的工程选择的版本对齐
	export fortify_version="18.20"
	#### ------------------------------------------------------------------


	### 以下内容固定不要动
	#### ------------------------------------------------------------------
	export CODEDEX_PATH=$gateboxPath/CodeDEX_V5
	export FORTIFY_HOME=$CODEDEX_PATH/tool/tools/fortify_${fortify_version}
	export ZA_HOME=$CODEDEX_PATH/tool/7za/Linux
	export PATH=$FORTIFY_HOME/bin:$PATH
	export FORTIFY_BUILD_ID=linus_yyds
	export inter_dir=$INTER_DIR
	export for_tmp_dir=$inter_dir/for_tmp
	export project_root=$SRC_WS/$SCAN_DIR
	#### ------------------------------------------------------------------

	### 以下内容固定不要动
	#### ------------------------------------------------------------------
	if [[ -d $inter_dir ]]
	then
	    rm -rf $inter_dir
	fi

	#Fortify Scan
	cd $project_root
	echo "98kar "$(find . -name "*.js" | wc -l)
	sourceanalyzer -b $FORTIFY_BUILD_ID -clean
	#### ------------------------------------------------------------------

	## $project_root这个参数说明
	### $project_root指的是代码仓的根目录
	# ------------------------------------------------------------------
	### 如果你的代码仓下面的文件是存放在隐藏目录下，如.config下
	### 则需显式的指明  "$project_root/.config/**/*.js"

	### 如果你的代码仓下面的文件隐藏文件，如.config.js下
	### 则需显式的指明该文件 $project_root/**/.config.js"
	# ------------------------------------------------------------------

	### 主要是靠以下两句抓取文件的
	#### 不要往上加vue这些，抓取不了
	#### ------------------------------------------------------------------
	sourceanalyzer -Xmx1G -b $FORTIFY_BUILD_ID -Dcom.fortify.sca.ProjectRoot=$for_tmp_dir "$project_root/**/*.js"
	sourceanalyzer -Xmx1G -b $FORTIFY_BUILD_ID -Dcom.fortify.sca.ProjectRoot=$for_tmp_dir "$project_root/**/*.ts"
	#### ------------------------------------------------------------------

	### 以下内容固定不要动
	#### ------------------------------------------------------------------
	sourceanalyzer -Xmx3G -b $FORTIFY_BUILD_ID -Dcom.fortify.sca.ProjectRoot=$for_tmp_dir -export-build-session $FORTIFY_BUILD_ID.mbs
	mv $FORTIFY_BUILD_ID.mbs $inter_dir

	cd $inter_dir
	$ZA_HOME/7za a -tzip fortify.zip $FORTIFY_BUILD_ID.mbs
	#### ------------------------------------------------------------------

library(ggplot2)
library(reshape2)

out.file <- "ch2_force_oacc_cuda.png"
## out.file <- "ch2_cpu_gpu_oacc.png"

data <- read.table("./profile.dat")

colnames(data) <- c("K40t\nOpenACC", "K40t\nCUDA", "P100\nOpenACC", "P100\nCUDA")
## colnames(data) <- c("E5-2680v3", "K40t\nOpenACC", "P100\nOpenACC")

data <- cbind(impl = rownames(data), data)
df <- melt(data)
g <- ggplot(
  df,
  aes (
    x = variable,
    y = value,
    fill= impl
  )
)
g <- g + geom_bar(position = "dodge", stat = "identity", with=0.8)
g <- g + xlab("") # x label is empty
g <- g + ylab("Elapsed Time [s]") # y label
g <- g + theme(text = element_text(size = 22)) # change text size
g <- g + labs(fill="") # remove legend title
g <- g + theme(axis.title.y = element_text(vjust = 0.28)) # 軸のラベル位置を調整
g <- g + theme(axis.text = element_text(color = "black")) # axis textの色を灰色から黒色に変更

plot(g)
ggsave(out.file)

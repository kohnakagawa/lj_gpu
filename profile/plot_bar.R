library(ggplot2)
library(reshape2)

## out.file <- "force_k40t_oacc_cuda.png"
## out.file <- "force_p100_oacc_cuda.png"
out.file <- "cpu_gpu_oacc.png"

data <- read.table("./profile.dat")
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

## g <- g + ggtitle("Tesla K40t") # add graph title
## g <- g + ggtitle("Tesla P100") # add graph title

plot(g)
ggsave(out.file)

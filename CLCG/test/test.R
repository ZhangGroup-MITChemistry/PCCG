require(pbs)
require(arrow)

x = seq(-pi, pi, length.out = 200)
knots = seq(-pi, pi, length.out = 8)
interior_knots = knots[c(-1, -8), drop = FALSE]
boundary_knots = knots[c(1, 8)]

degree = 3
#design_matrix = bs(x, knots = interior_knots, Boundary.knots = boundary_knots, intercept = FALSE)
design_matrix = pbs(x, knots = interior_knots, Boundary.knots = boundary_knots, intercept = FALSE)

design_matrix = as.data.frame(design_matrix[,])

write_feather(design_matrix, "design_matrix_bs_R.feather")


#pdf("./bs_design_R.pdf")
#
#for (j in seq(ncol(design_matrix))){
#    plot(x, design_matrix[,j])
#    par(new=TRUE)    
#}
#dev.off()
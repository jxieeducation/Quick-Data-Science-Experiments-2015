package com.linear

import scala.io.Source

class LinearRegression(xs: Seq[Double], ys: Seq[Double]){
	require(xs.size == ys.size, "xs and ys must be same length")

  	def mean(values: Seq[Double]): Double = values.sum / values.size

  	def slope: Double = {
    	val xmean = mean(xs)
    	val ymean = mean(ys)

    	val numerator = xs.zip(ys).foldLeft(0.0) { case (sum, (x, y)) =>
      		sum + ((x - xmean) * (y - ymean))
    	}
    	val denominator = xs.foldLeft(0.0) { (sum, x) =>
      		sum + math.pow(x - xmean, 2)
    	}

		numerator / denominator
  	}

  	def yIntercept: Double = mean(ys) - (slope * mean(xs))
}


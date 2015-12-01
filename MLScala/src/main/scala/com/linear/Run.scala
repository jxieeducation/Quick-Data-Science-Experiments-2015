import scala.io.Source

package com.linear {
    object RunLinear extends App {
        val xs = Seq(1.0, 2.0, 3.0)
        val ys = Seq(3.0, 5.0, 7.0)

        val linearModel = new LinearRegression(xs, ys)
        println("Model generated with")
        println(s"Slope: ${linearModel.slope}")
        println(s"Y-Intercept: ${linearModel.yIntercept}")
        println("\n")
        println("Estimated Linear Model:")
        println(s"Y = ${linearModel.yIntercept} + (${linearModel.slope} * X)")
    }
}

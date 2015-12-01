object P32{

	def gcd(n1: Int, n2: Int): Int = {
		var smaller: Int = if (n1 < n2) n1 else n2

		var gcd:Int = 1
		var count:Int = 2
		while(count <= smaller){
			if (n1 % count == 0 && n2 % count == 0){
				gcd = count
			}
			count += 1
		}

		return gcd
	}

	def main(args: Array[String]) {
    	println(gcd(10, 130))
    	println(gcd(4, 6))
    	println(gcd(130, 260))
	}
}

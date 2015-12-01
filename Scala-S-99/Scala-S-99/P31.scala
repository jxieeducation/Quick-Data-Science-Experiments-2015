object P31{
	def isPrime(n:Int): Boolean = {
		var count:Int = 2
		while(count < n / 2){
			if (n % count == 0){
				return false
			}
			count += 1
		}
		true
	}

	def main(args: Array[String]) {
    	println(isPrime(15))
    	println(isPrime(31))
    	println(isPrime(22))
    	println(isPrime(1))
	}
}

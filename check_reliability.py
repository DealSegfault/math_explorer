"""Offline regression checks: python3 check_reliability.py (temporary data only)."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor
from threading import Event


def main():
    with tempfile.TemporaryDirectory(prefix="math-solver-check-") as directory:
        os.environ["MATH_EXPLORER_DATA_DIR"] = directory
        from backends.sympy_backend import SymPyBackend
        from backends.z3_backend import Z3Backend
        from verification.ensemble import VerificationEnsemble
        from verification.expressions import parse_expression
        from cache_manager import cache
        from fastapi.testclient import TestClient
        import server

        sympy, z3, verifier = SymPyBackend(), Z3Backend(), VerificationEnsemble()
        problems = [
            ("Count all integers n < 1000 such that n is divisible by 6, not divisible by 4, and not divisible by 9.", "55"),
            ("How many positive integer divisors of 2024 are multiples of 4? Note that 2024 = 2^3 * 11 * 23.", "8"),
            ("Compute the Legendre symbol (11/13) using the Law of Quadratic Reciprocity.", "-1"),
            ("Calculate the Legendre symbol (7/17) using Gauss's Law of Quadratic Reciprocity step by step.", "-1"),
            ("In the finite field F_13 with 13 elements, determine the number of nonzero elements that are cubic residues (cubes of nonzero elements).", "4"),
            ("Find the number of positive integers n <= 100 such that the polynomial x^2 + x + 1 divides x^(2n) + 1 in R[x].", "0"),
            ("Compute 1 / 2 + 1 / 3", "5/6"),
        ]
        for query, answer in problems:
            result = sympy.solve(query)
            assert result["is_exact"] and result["extracted_answer"] == answer, (query, result)
            assert verifier.verify(result["solution"], ground_truth=answer, evidence=result.get("verification_evidence")).is_verified, query
        for query in ["Compute 1/3 modulo 7", "What is the number of divisors of 36 plus 2+2?",
                      "Find n <= 10 such that x^2 + x + 1 divides x^3 - 1",
                      "Count integers n < 100 divisible by 6 and prime",
                      "How many positive integer divisors of 0 are multiples of 4?",
                      "Compute the Legendre symbol (3/4)", "Compute 1 / 0"]:
            assert not sympy.solve(query)["is_exact"], query
        for comparison, bound, expected in [("<", 12, 1), ("<=", 12, 2), ("<", 0, 0)]:
            query = f"Count integers n {comparison} {bound} divisible by 6"
            assert sympy.solve(query)["extracted_answer"] == str(expected)
            assert z3.solve(query)["extracted_answer"] == str(expected)
        assert z3._parse_divisibility_sieve(problems[0][0]) == (999, [6], [4, 9])
        assert z3.solve(problems[0][0])["extracted_answer"] == "55"
        assert z3._parse_divisibility_sieve("Count integers divisible by 6") is None
        assert z3._parse_divisibility_sieve("Count integers n < 10 divisible by 0") is None
        counterexample = z3.find_counterexample(['x'], 'Eq(Mod(x,51),x)', {'x': (1, 51)})
        assert counterexample['counterexample_found'] and counterexample['model']['x'] == 51

        assert verifier.verify('1+1 = 2\nFinal answer: \\boxed{999}').status == 'UNVERIFIED'
        assert verifier.verify('1+1 = 2 = 3\n\\boxed{3}', ground_truth='3').status == 'REFUTED'
        assert verifier.verify('x+1 = 3\n\\boxed{2}', ground_truth='2').status != 'REFUTED'
        assert verifier.verify(r'\boxed{\frac{1}{2}}', ground_truth='1/2').is_verified
        assert verifier.verify(r'\boxed{\frac{1}{2}}', ground_truth='2').status == 'REFUTED'
        assert verifier.extract_boxed_answer(r'\boxed{1} then \boxed{\frac{1}{2}') is None
        assert verifier.check_cas_equality(r'\left(1+2\right)*3', '9')[0] is True
        assert verifier.check_cas_equality('2^3+1', '9')[0] is True
        assert verifier.check_z3_congruence('1/2', '0', '3')[0] is None
        assert verifier.check_z3_congruence('5', '2', '3')[0] is True
        with patch('os.system') as system:
            for expression in ["__import__('os').system('echo forbidden')", 'x.__class__',
                               '2**100000000', 'sqrt(-1)/0', '[1, 2]', '1e999']:
                assert verifier.check_cas_equality(expression, '0')[0] is None, expression
            system.assert_not_called()
            assert z3.find_counterexample(['x'], "__import__('os').system('echo forbidden')", {'x': (1, 2)})['sat'] is None
            system.assert_not_called()
        assert str(parse_expression('1/2 + 1/3')) == '5/6'
        assert cache.hash_key('ab', 'c') != cache.hash_key('a', 'bc')
        assert cache.hash_key('x', 'X') != cache.hash_key('X', 'x')
        assert cache.hash_key(None) != cache.hash_key('None')

        harness = server.harness
        with patch.object(harness, '_router', None):
            result = harness.explore('Compute 1 / 2 + 1 / 3')
            assert result['is_verified'] and result['early_exit']
            forced = harness.explore('Compute 1 / 2 + 1 / 3', engine='sympy_cas')
            assert forced['is_verified'] and forced['solver_engine'] == 'sympy_cas'
            assert harness._router is None, 'Exact solves must not initialize external routing'
        nodes = harness.gm.get_graph_data()['nodes']
        assert any(n['type'] == 'sympy_proof' and n['data']['engine'] == 'sympy_cas' for n in nodes)
        assert Path(harness.gm.filepath).parent == Path(directory)

        with TestClient(server.app) as client:
            assert client.get('/').status_code == 200
            assert client.post('/api/explore', json={'query': '  '}).status_code == 400
            for invalid in ({'query': '1+1', 'max_tokens': 0}, {'query': '1+1', 'engine': 'unknown'},
                            {'query': '1+1', 'top_k_nodes': -1}):
                assert client.post('/api/explore', json=invalid).status_code == 422
            assert client.post('/api/rrsi/loop', json={'steps': -1}).status_code == 422
            response = client.post('/api/explore', json={'query': 'Compute 10 / 4'})
            assert response.status_code == 200, response.text
            assert response.json()['is_verified']
            started, release = Event(), Event()
            def slow_explore(**kwargs):
                started.set()
                assert release.wait(5), 'Test failed to release slow solver'
                return {'solution': 'done'}
            with patch.object(harness, 'explore', side_effect=slow_explore):
                with ThreadPoolExecutor(max_workers=1) as pool:
                    pending = pool.submit(client.post, '/api/explore', json={'query': 'slow'})
                    try:
                        assert started.wait(3)
                        assert client.get('/api/graph').status_code == 200
                        assert client.get('/api/harness').status_code == 200
                        assert client.post('/api/explore', json={'query': 'overlap'}).status_code == 409
                    finally:
                        release.set()
                    assert pending.result(timeout=3).status_code == 200
        print('PASS: exact math, false proofs, safe parsing, cache isolation, graph provenance, API validation and concurrent reads.')


if __name__ == '__main__':
    main()
